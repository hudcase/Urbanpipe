import os
import json

dir_base_path = "../dataset/video_label_file/"
output_base_path = "./dataset/"

for split in ["train", "val"]:
    json_file = f"video_{split}.json"
    json_path = os.path.join(dir_base_path, json_file)

    out_json_file = f"data_vl_{split}.jsonl"
    out_json_path = os.path.join(output_base_path, out_json_file)

    # 读取原始 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        js_datas = json.load(f)

    count = 0

    # 写入 JSONL
    with open(out_json_path, 'w', encoding='utf-8') as f_out:
        for k in js_datas:
            dt = {}
            dt["id"] = k
            super_image_name = f"{os.path.splitext(k)[0]}_superImage.jpg"
            dt["image_path"] = super_image_name
            dt["labels"] = js_datas[k]

            # 写入一行 JSON
            f_out.write(json.dumps(dt, ensure_ascii=False) + "\n")
            count += 1

    print(f"{out_json_file} 已生成，共 {count} 条数据")