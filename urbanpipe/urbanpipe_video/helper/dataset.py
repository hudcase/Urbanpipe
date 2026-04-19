import os
import json
import torch
import numpy as np
import av

from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor

class JSON_VideoDataset(Dataset):
    """
    基于 JSON 标注文件的视频分类数据集

    功能流程：

        1. 从 JSON 文件读取视频路径与标签
        2. 使用 PyAV 解码视频
        3. 进行随机等间隔帧采样
        4. 使用 HuggingFace ImageProcessor 做预处理
        5. 返回 (T, C, H, W) 视频张量 + multi-label one-hot 标签

    返回格式：

        video_tensor: (T, C, H, W)
        labels_tensor: (num_classes,)
    """

    def __init__(
        self,
        root,
        json_file,
        image_processor,
        num_classes=17,
        num_frames=8,
    ):
        """
        参数说明：

            root:
                视频文件所在根目录

            json_file:
                JSON 文件路径
                格式示例：
                {
                    "video1.mp4": [0, 3],
                    "video2.mp4": [2]
                }

            image_processor:
                HuggingFace ImageProcessor
                用于图像 resize / normalize 等预处理

            num_classes:
                分类总类别数

            num_frames:
                每个视频采样的帧数 T
        """

        # 基本参数保存
        self.root = root
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.image_processor = image_processor

        # 读取 JSON 标签文件
        with open(json_file, "r") as f:
            js_datas = json.load(f)

        # 保存视频路径和标签
        self.video_paths = []
        self.video_labels = []

        for video_name, label in js_datas.items():

            # 拼接完整视频路径
            video_path = os.path.join(root, video_name)

            self.video_paths.append(video_path)
            self.video_labels.append(label)

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.video_paths)

    # ======================================================
    # 帧采样函数
    # ======================================================
    def sample_random_uniform_indices(self, total_frames):
        """
        随机起点 + 等间隔时间采样

        采样思想：

            将整个视频时间轴均匀划分为 num_frames 个区间，
            每个区间采样 1 帧。

            同时在第一个区间内随机选择起点，
            使每次训练时采样位置不同，提高泛化能力。

        参数：
            total_frames: 视频总帧数

        返回：
            indices: shape (num_frames,)
        """

        clip_len = self.num_frames

        # ---------------------------------------------
        # 情况1：视频帧数少于采样帧数
        # ---------------------------------------------
        # 此时无法真正等间隔采样
        # 使用 linspace 均匀分布（可能会重复最后一帧）
        if total_frames <= clip_len:

            indices = np.linspace(
                0,
                total_frames - 1,
                clip_len
            )

            # 防止浮点误差越界
            return np.clip(
                indices,
                0,
                total_frames - 1
            ).astype(np.int64)

        # ---------------------------------------------
        # 情况2：正常视频长度
        # ---------------------------------------------

        # 每段时间长度（保持 float 更精确）
        interval = total_frames / clip_len

        # 在第一个区间内随机起点
        # 例如 interval = 12.5
        # start ∈ [0, 12.5)
        start = np.random.uniform(0, interval)

        # 生成等间隔索引
        # int() 相当于向下取整 floor
        indices = np.array([
            int(start + interval * i)
            for i in range(clip_len)
        ])

        # 防止最后一个 index 超出 total_frames-1
        return np.clip(
            indices,
            0,
            total_frames - 1
        ).astype(np.int64)

    # ======================================================
    # 视频解码函数
    # ======================================================
    def read_video_pyav(self, container, indices):
        """
        使用 PyAV 解码指定帧

        container:
            av.open() 返回的容器

        indices:
            需要解码的帧索引数组

        返回：
            video: (T, H, W, 3) numpy array
        """

        frames = []

        # 从视频开头开始解码
        # 非常重要，否则 decode 位置不确定
        container.seek(0)

        start_index = indices[0]
        end_index = indices[-1]

        # 转成 set，提升查找效率
        # 原本 i in indices 是 O(n)
        # 现在 i in set 是 O(1)
        indices_set = set(indices)

        # 逐帧解码
        for i, frame in enumerate(container.decode(video=0)):

            # 超过最大索引就停止
            if i > end_index:
                break

            # 只保存需要的帧
            if i >= start_index and i in indices_set:

                # 转换为 RGB numpy array
                frames.append(
                    frame.to_ndarray(format="rgb24")
                )

        # 堆叠为 (T, H, W, 3)
        return np.stack(frames)

    # ======================================================
    # 核心数据读取函数
    # ======================================================
    def __getitem__(self, idx):
        """
        返回一个样本：

            video_tensor: (T, C, H, W)
            labels_tensor: (num_classes,)
        """

        video_path = self.video_paths[idx]
        labels = self.video_labels[idx]

        # 打开视频容器
        container = av.open(video_path)

        try:
            # 获取视频总帧数
            total_frames = container.streams.video[0].frames

            # 有些视频 metadata 记录不准确
            # 如果为 0，需要手动统计
            if total_frames == 0:
                total_frames = sum(
                    1 for _ in container.decode(video=0)
                )
                container.seek(0)

            # 时间采样
            indices = self.sample_random_uniform_indices(
                total_frames
            )

            # 解码指定帧
            video = self.read_video_pyav(
                container,
                indices
            )

        finally:
            # 无论是否报错，都关闭容器
            # 防止内存泄漏
            container.close()

        # --------------------------------------------------
        # 图像预处理
        # --------------------------------------------------
        # ImageProcessor 接受 list[np.ndarray]
        # 输出:
        # pixel_values: (1, T, C, H, W)
        inputs = self.image_processor(
            list(video),
            return_tensors="pt",
        )

        # 去掉 batch 维
        # 得到 (T, C, H, W)
        video_tensor = inputs["pixel_values"].squeeze(0)

        # --------------------------------------------------
        # multi-label one-hot 编码
        # --------------------------------------------------
        labels_tensor = torch.zeros(
            self.num_classes,
            dtype=torch.float32
        )

        for label in labels:
            labels_tensor[label] = 1

        return video_tensor, labels_tensor


def get_dataset_urbanpipe(args):
    # 加载 TimeSformer 官方图像预处理器
    # 来自 facebook/timesformer-base-finetuned-k400
    #
    # 作用：对视频执行预处理：
    #
    # 1. Resize
    #    将图像缩放到模型期望尺寸（通常 shortest_edge=256）
    #
    # 2. Center Crop
    #    中心裁剪为 224 x 224
    #
    # 3. 转为 Tensor
    #    (H, W, C) uint8
    #      → (C, H, W) float32
    #      → 并归一化到 [0,1]
    #
    # 4. Normalize 标准化
    #    使用 Kinetics-400 训练时的均值和标准差：
    #    mean = [0.45, 0.45, 0.45]
    #    std  = [0.225, 0.225, 0.225]
    #
    # 5. 输出格式
    #    输入:  list of frames [(H,W,C), ...]
    #    输出:  tensor (1, T, C, 224, 224)
    #
    #    在 Dataset 中 squeeze(0) 后：
    #           (T, C, 224, 224)

    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/timesformer-base-finetuned-k400"
    )

    train_dataset = JSON_VideoDataset(
        root=args.base_path,
        json_file=args.label_file_train,
        image_processor=image_processor,
        num_classes=args.num_classes,
        num_frames=8,
    )

    test_dataset = JSON_VideoDataset(
        root=args.base_path,
        json_file=args.label_file_val,
        image_processor=image_processor,
        num_classes=args.num_classes,
        num_frames=8,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader
