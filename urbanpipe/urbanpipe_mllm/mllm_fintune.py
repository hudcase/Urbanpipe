"""
Qwen2-VL 多模态大模型微调与推理脚本
实现基于LoRA的高效参数微调，适用于视觉-语言多标签分类任务
"""

import torch
import json
import re
import os
from datasets import Dataset
from modelscope import AutoTokenizer
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import os
from utils import CLASS_NAMES, ID2CLASS, CLASS2ID, SYSTEM_PROMPT, normalize_label, predict, parse_prediction, predict_and_parse

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def process_func(example):
    """
    多模态数据预处理函数
    """
    MAX_LENGTH = 8192

    output_content = example['labels']

    # 提取图像路径
    file_path = example['image_path']
    file_path = os.path.join("./dataset/superImage", file_path)

    # 标准化输出标签
    normalized_output = normalize_label(output_content)

    # 构建多模态对话模板
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": file_path,
                "resized_height": 518,
                "resized_width": 518
            },
            {
                "type": "text",
                "text": SYSTEM_PROMPT
            },
        ]
    }]

    # 应用模板生成结构化文本
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 处理视觉输入
    image_inputs, video_inputs = process_vision_info(messages)

    # 多模态特征整合
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.tolist() for k, v in inputs.items()}

    # 使用标准化后的标签作为输出
    response = tokenizer(normalized_output, add_special_tokens=False)

    # 构建完整输入序列
    input_ids = inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]

    # 标签：指令部分掩码为-100
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 长度截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": torch.tensor(inputs['pixel_values']),
        "image_grid_thw": torch.tensor(inputs['image_grid_thw']).squeeze(0)
    }


# ===================== 主程序 =====================

if __name__ == "__main__":

    # ==================== 初始化阶段 ====================
    MODEL_PATH = "./Qwen/Qwen2-VL-2B-Instruct/"
    OUTPUT_DIR = "./output/Qwen2-VL-2B-MultiLabel"
    RESULT_JSONL_PATH = os.path.join(OUTPUT_DIR, "prediction_results.jsonl")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/5] 加载分词器和处理器...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("[2/5] 加载基础模型...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    # ==================== 数据处理阶段 ====================
    print("[3/5] 处理训练数据...")

    # import pdb;pdb.set_trace()
    train_ds = Dataset.from_json("./dataset/data_vl_train.jsonl")
    print(f"  训练样本数: {len(train_ds)}")
    print(f"  样本字段: {train_ds.column_names}")

    first_sample = train_ds[0]
    raw_output = first_sample["labels"]
    normalized = normalize_label(raw_output)
    print(f"  原始标签示例: {raw_output}")
    print(f"  标准化标签:   {normalized}")

    train_dataset = train_ds.map(process_func)
    print(f"  预处理完成，特征字段: {train_dataset.column_names}")

    # ==================== 模型配置阶段 ====================
    print("[4/5] 配置LoRA...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj",
            "up_proj", "down_proj"
        ],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.01,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"  可训练参数: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # ==================== 训练阶段 ====================
    print("[5/5] 开始训练...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=20,
        save_steps=2,
        save_total_limit=5,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=4,
        bf16=True,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True
        ),
    )

    trainer.train()

    final_save_path = os.path.join(OUTPUT_DIR, "final")
    peft_model.save_pretrained(final_save_path)
    print(f"  模型已保存到: {final_save_path}")
# CUDA_VISIBLE_DEVICES=1 python /search_sc_ssd/qixiaohua03/project/class/urbanpipee/mllm_fintune.py