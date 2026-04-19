## 训练命令示例
```bash
python3 ./train.py \
    --base_path ../dataset/tony_data_video/ \
    --label_file_train ../dataset/video_label_file/video_train.json \
    --label_file_val ../dataset/video_label_file/video_val.json \
    --model_name timesformer \
    --learning_rate 0.001 \
    --epochs 10
```

## timeSFormer 模型学习代码

python3 helper/TimeSformer.py