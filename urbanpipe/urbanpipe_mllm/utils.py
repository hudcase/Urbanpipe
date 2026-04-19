
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

CLASS_NAMES = [
    "正常", "沉积", "腐蚀", "障碍物", "结垢", "树根",
    "破裂", "变形", "异物穿入", "浮渣", "脱节", "暗接",
    "残墙", "渗漏", "起伏", "错口", "脱落"
]

ID2CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS2ID = {name: i for i, name in enumerate(CLASS_NAMES)}

MAX_LABELS = 5


SYSTEM_PROMPT = (
    "你是一个管道检测专家，需要根据图像识别管道内部缺陷。\n\n"

    "输入说明：\n"
    "输入图片由同一视频中等间隔抽取的四帧组成，并按时间顺序拼接成一张大图。\n"
    "请同时观察四个子图，检测可能存在的缺陷。\n\n"

    "缺陷类别共有17种：\n"
    "0 正常\n"
    "1 沉积\n"
    "2 腐蚀\n"
    "3 障碍物\n"
    "4 结垢\n"
    "5 树根\n"
    "6 破裂\n"
    "7 变形\n"
    "8 异物穿入\n"
    "9 浮渣\n"
    "10 脱节\n"
    "11 暗接\n"
    "12 残墙\n"
    "13 渗漏\n"
    "14 起伏\n"
    "15 错口\n"
    "16 脱落\n\n"

    "分析方法，一步一步思考（在内部思考，不要输出）：\n"
    "1. 依次观察四个子图。\n"
    "2. 判断每个子图是否存在缺陷。\n"
    "3. 汇总所有子图中的缺陷类别。\n"
    "4. 去重。\n"
    "5. 按类别编号从小到大排序。\n\n"

    "Few-shot 示例：\n\n"

    "示例1：\n"
    "图像情况：管道内部光滑，没有明显异常。\n"
    "输出：\n"
    '{"缺陷": [0]}\n\n'

    "示例2：\n"
    "图像情况：管道底部有沉积，同时管壁存在腐蚀。\n"
    "输出：\n"
    '{"缺陷": [1, 2]}\n\n'

    "示例3：\n"
    "图像情况：管道出现树根侵入，并伴随沉积。\n"
    "输出：\n"
    '{"缺陷": [1, 5]}\n\n'

    "输出规则：\n"
    "1. 每张图最多输出5种缺陷。\n"
    "2. 如果没有发现缺陷，请返回 {\"缺陷\": [0]}。\n"
    "3. 只允许输出JSON，不要输出解释、文字或代码块。\n"
    "4. 返回格式必须严格如下：\n"
    '{"缺陷": [类别编号1, 类别编号2, ...]}'
)
# ===================== 数据处理工具函数 =====================

def normalize_label(raw_label_str):
    """
    将原始标签字符串标准化为固定格式的JSON字符串
    """
    try:
        if isinstance(raw_label_str, str):
            data = json.loads(raw_label_str)
        else:
            data = raw_label_str

        if isinstance(data, dict):
            labels = data.get("缺陷") or data.get("defects") or data.get("labels") or []
        elif isinstance(data, list):
            labels = data
        else:
            labels = [0]

        valid_labels = sorted(set(int(l) for l in labels if 0 <= int(l) <= 16))
        valid_labels = valid_labels[:MAX_LABELS]

        if not valid_labels:
            valid_labels = [0]

    except (json.JSONDecodeError, ValueError, TypeError):
        valid_labels = [0]

    return json.dumps({"缺陷": valid_labels}, ensure_ascii=False)

# ===================== 推理函数 =====================

def predict(messages, model):
    """
    多模态推理函数

    参数：
        messages: list - 多模态对话数据
        model: PreTrainedModel - 模型实例

    返回：
        str - 生成的文本响应
    """
    # 构建对话模板
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
    ).to("cuda")

    # 生成响应
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

    # 去除输入部分
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # 解码
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def parse_prediction(output_text):
    """
    解析模型输出，提取缺陷类别

    参数：
        output_text: str - 模型生成的文本

    返回：
        label_ids: list[int] - 缺陷类别编号列表
        label_names: list[str] - 缺陷类别名称列表
    """
    try:
        # 尝试直接解析JSON
        data = json.loads(output_text.strip())
        if isinstance(data, dict):
            labels = data.get("缺陷", [0])
        elif isinstance(data, list):
            labels = data
        else:
            labels = [0]
    except json.JSONDecodeError:
        # 尝试用正则提取数字
        numbers = re.findall(r'\d+', output_text)
        labels = [int(n) for n in numbers if 0 <= int(n) <= 16]
        if not labels:
            labels = [0]

    # 确保标签是整数
    valid_labels = []
    for l in labels:
        try:
            val = int(l)
            if 0 <= val <= 16:
                valid_labels.append(val)
        except (ValueError, TypeError):
            continue

    if not valid_labels:
        valid_labels = [0]

    # 去重、排序、截断
    valid_labels = sorted(set(valid_labels))[:MAX_LABELS]

    # 转换为名称
    names = [ID2CLASS.get(l, f"未知({l})") for l in valid_labels]

    return valid_labels, names


def predict_and_parse(messages, model):
    """
    推理并解析结果

    返回：
        dict - 包含原始输出、类别编号、类别名称
    """
    raw_output = predict(messages, model)
    label_ids, label_names = parse_prediction(raw_output)

    return {
        "raw_output": raw_output,
        "label_ids": label_ids,
        "label_names": label_names
    }

