# 项目说明

## 1️⃣ 视频抽帧

从每个视频中**平均抽取 16 帧**，并按视频名创建文件夹保存。

### 命令

```bash
python3 ./tools/video2images.py \
--input_folder ../dataset/tony_data_video \
--output_folder ./dataset/tony_data_videoFrame
```

### 参数说明
* 注意`../dataset/`和`./dataset/`不是一个位置
* `--input_folder`：视频文件夹路径
* `--output_folder`：抽取帧保存路径

---

## 2️⃣ 模型训练

```bash
python3 ./train.py \
--base_path ./dataset/tony_data_videoFrame \
--label_file_train ../dataset/video_label_file/video_train.json \
--label_file_val ../dataset/video_label_file/video_val.json \
--model_name resnet18 \
--batch_size 24 \
--learning_rate 0.001 \
--epochs 20
```
### 参数说明

* `--base_path`：视频帧数据目录
* `--label_file_train`：训练标签 JSON
* `--label_file_val`：验证标签 JSON
* `--model_name`：模型名称,支持：ResNet 系列：resnet18、resnet34、resnet50等；ViT 系列：vit_b_16、vit_b_32、vit_l_16
* `--batch_size`：batch size
* `--learning_rate`：学习率
* `--epochs`：训练轮数
