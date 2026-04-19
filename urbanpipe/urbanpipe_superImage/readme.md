# 项目说明

## 1️⃣ 视频抽帧

从每个视频中**平均抽取 9 帧**，并按视频名创建文件夹保存。

### 命令
cd urbanpipe_superImage
```bash
python3 ./tools/video2superImage.py
```
从一个文件夹中的多个视频里自动抽取关键帧，并将每个视频生成一张九宫格拼接图像。

整体逻辑可以分为两个步骤：

第一步是从视频中抽取代表性帧。
程序会遍历指定文件夹中的所有 mp4 视频文件。对于每个视频，它会读取视频的总帧数，并在视频时间轴上均匀选取 9 个位置的帧进行提取。

第二步是生成九宫格图像。
程序会读取刚刚提取出的 9 张帧图像，并将它们统一缩放，然后按三行三列的方式拼接成一张大图。
---

## 2️⃣ 模型训练

```bash
python3 ./train.py \
--base_path ./dataset/superImage \
--label_file_train ../dataset/video_label_file/video_train.json \
--label_file_val ../dataset/video_label_file/video_val.json \
--model_name vit_h_14 \
--batch_size 24 \
--learning_rate 0.001 \
--epochs 20
```
### 参数说明

* `--base_path`：视频帧数据目录
* `--label_file_train`：训练标签 JSON
* `--label_file_val`：验证标签 JSON
* `--model_name`：模型名称,支持：ViT 系列：vit_h_14
* `--batch_size`：batch size
* `--learning_rate`：学习率
* `--epochs`：训练轮数
