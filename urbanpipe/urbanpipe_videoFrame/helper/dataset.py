import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

class JSON_ImageDataset(Dataset):
    def __init__(self, root, json_file, transform=None, num_classes=3):
        """
        初始化函数，设置数据集的根目录、JSON文件、转换方式和类别数量。
        
        Args:
        - root (str): 数据集的根目录路径。
        - json_file (str): 包含标签信息的JSON文件路径。
        - transform (torchvision.transforms.Compose, optional): 图像转换操作。默认为None。
        - num_classes (int): 类别数量。默认为3。
        """
        self.root = root
        self.transform = transform        
        self.num_classes = num_classes

        # 读取JSON文件中的数据
        with open(json_file, 'r') as f:
            js_datas = json.load(f)

        image_paths = []
        image_labels = []
        for video_name in js_datas:
            label = js_datas[video_name]
            image_dir_name = video_name.replace(".mp4", "")
            for index in range(16):
                image_path = os.path.join(self.root, image_dir_name, f"frame_{index}.jpg")
                image_paths.append(image_path)
                image_labels.append(label)
        self.image_paths = image_paths
        self.image_labels = image_labels

    def __len__(self):
        """
        返回数据集的长度。
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取数据集中索引为idx的图像和对应的标签。
        
        Args:
        - idx (int): 图像的索引。
        
        Returns:
        - image (torch.Tensor): 转换后的图像张量。
        - labels_tensor (torch.Tensor): 标签张量，one-hot编码的形式。
        """
        image_path, labels = self.image_paths[idx], self.image_labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels_tensor = torch.zeros(self.num_classes)  # 创建一个全零张量作为标签
        for label in labels:
            labels_tensor[label] = 1  # 将标签对应的位置置为1
        return image, labels_tensor

def get_dataset_urbanpipe(args):
    """
    获取UrbanPipe数据集的训练和测试数据加载器。
    
    Args:
    - args (namespace): 参数名称空间，包含必要的参数信息。
    
    Returns:
    - train_loader (torch.utils.data.DataLoader): 训练数据加载器。
    - test_loader (torch.utils.data.DataLoader): 测试数据加载器。
    """
    # 定义训练数据的图像转换操作
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 定义测试数据的图像转换操作
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 加载训练数据集
    train_dataset = JSON_ImageDataset(root=args.base_path, json_file=args.label_file_train, transform=train_transform, num_classes=args.num_classes)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4)

    # 加载测试数据集
    test_dataset = JSON_ImageDataset(root=args.base_path, json_file=args.label_file_val, transform=val_transform, num_classes=args.num_classes)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    
    return train_loader, test_loader

