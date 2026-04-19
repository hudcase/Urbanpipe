"""
Qwen2-VL 多模态大模型推理与评估脚本
只计算最简单的 binary mAP
"""

import torch
import json
import os
import numpy as np

from modelscope import AutoTokenizer
from peft import PeftModel
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)

from utils import (
    CLASS_NAMES, ID2CLASS,
    SYSTEM_PROMPT,
    normalize_label,
    parse_prediction,
    predict_and_parse
)

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

NUM_CLASSES = len(CLASS_NAMES)


# =========================================================
# multi-hot
# =========================================================

def labels_to_binary_vector(label_ids, num_classes=NUM_CLASSES):
    vec = np.zeros(num_classes, dtype=np.float32)

    for lid in label_ids:
        if 0 <= lid < num_classes:
            vec[lid] = 1

    return vec


# =========================================================
# AP
# =========================================================

def compute_ap_binary(y_true, y_pred):
    """
    二值预测 AP

    AP = Precision × Recall
    """

    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    ap = precision * recall
    return ap


# =========================================================
# mAP
# =========================================================

def compute_map_binary(predictions, ground_truths):

    y_true = np.array([
        labels_to_binary_vector(x) for x in ground_truths
    ])

    y_pred = np.array([
        labels_to_binary_vector(x) for x in predictions
    ])

    ap_per_class = {}

    for c in range(NUM_CLASSES):

        ap = compute_ap_binary(
            y_true[:, c],
            y_pred[:, c]
        )

        cls_name = ID2CLASS[c]

        ap_per_class[cls_name] = round(float(ap), 4)

    valid_classes = [
        c for c in range(NUM_CLASSES)
        if y_true[:, c].sum() > 0
    ]
    # import pdb;pdb.set_trace()
    if len(valid_classes) == 0:
        mAP = 1.0
    else:
        mAP = np.mean([
            ap_per_class[ID2CLASS[c]]
            for c in valid_classes
        ])

    return round(float(mAP), 4), ap_per_class


# =========================================================
# evaluate
# =========================================================

def evaluate_multilabel(predictions, ground_truths):

    mAP, ap_per_class = compute_map_binary(
        predictions,
        ground_truths
    )

    results = {
        "mAP": mAP,
        "AP_per_class": ap_per_class
    }

    return results


# =========================================================
# print
# =========================================================

def print_evaluation_report(results):

    print("\n================ Evaluation ================")

    print(f"\nmAP: {results['mAP']:.4f}\n")

    print("AP per class:")

    for cls, ap in results["AP_per_class"].items():
        print(f"{cls:10s}: {ap:.4f}")

    print("============================================\n")


# =========================================================
# main
# =========================================================

if __name__ == "__main__":

    MODEL_PATH = "/root/autodl-tmp/Qwen/Qwen2-VL-2B-Instruct/"
    CHECKPOINT_PATH = "./output/Qwen2-VL-2B-MultiLabel/checkpoint-4"
    OUTPUT_DIR = "./output/Qwen2-VL-2B-MultiLabel"

    RESULT_JSONL_PATH = os.path.join(
        OUTPUT_DIR,
        "prediction_results.jsonl"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================
    # load model
    # =========================================================

    print("[1/4] 加载模型...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH
    )

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        model_id=CHECKPOINT_PATH
    )

    model.eval()

    print("模型加载完成")


    # =========================================================
    # load dataset
    # =========================================================

    print("[2/4] 加载测试数据...")

    test_samples = []

    with open("./dataset/data_vl_train.jsonl", "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:
                test_samples.append(json.loads(line))

    print(f"测试样本数: {len(test_samples)}")


    # =========================================================
    # inference
    # =========================================================

    print("[3/4] 批量推理...")

    all_predictions = []
    all_ground_truths = []

    with open(RESULT_JSONL_PATH, "w", encoding="utf-8") as f_out:

        for idx, sample in enumerate(test_samples):

            img_path = os.path.join(
                "./dataset",
                sample["image_path"]
            )

            sample_id = sample.get(
                "id",
                os.path.splitext(os.path.basename(img_path))[0]
            )

            # ---------- ground truth ----------

            gt_raw = sample["labels"]

            gt_norm = normalize_label(gt_raw)

            gt_ids, gt_names = parse_prediction(gt_norm)

            # ---------- build input ----------

            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                        "resized_height": 518,
                        "resized_width": 518
                    },
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    }
                ]
            }]

            # ---------- inference ----------

            try:

                result = predict_and_parse(
                    messages,
                    model
                )

                pred_ids = result["label_ids"]
                pred_names = result["label_names"]
                raw_output = result["raw_output"]

            except Exception as e:

                print(f"推理失败: {sample_id}")

                pred_ids = [0]
                pred_names = ["正常"]
                raw_output = ""

            all_predictions.append(pred_ids)
            all_ground_truths.append(gt_ids)

            record = {
                "id": sample_id,
                "image_path": img_path,
                "ground_truth": gt_names,
                "prediction": pred_names,
                "raw_output": raw_output
            }

            f_out.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            if (idx + 1) % 10 == 0:
                print(f"{idx+1}/{len(test_samples)}")


    print("预测完成")


    # =========================================================
    # evaluation
    # =========================================================

    print("[4/4] 评估...")

    results = evaluate_multilabel(
        all_predictions,
        all_ground_truths
    )

    print_evaluation_report(results)