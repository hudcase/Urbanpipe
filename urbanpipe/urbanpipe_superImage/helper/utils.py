import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import random
import numpy as np

def calculate_mAP(y_true, y_score):
    """
    计算平均精度(mAP)。

    参数:
        y_true (numpy.ndarray): 真实标签数组。
        y_score (numpy.ndarray): 预测概率数组。

    返回:
        float: 平均精度。
    """
    # import pdb;pdb.set_trace()

    mAP = 0
    for i in range(y_true.shape[1]):
        mAP += average_precision_score(y_true[:, i], y_score[:, i])
    mAP /= y_true.shape[1]
    return mAP

def train(model, trainloader, criterion, optimizer, epoch, device):
    """
    训练函数，用于训练神经网络模型。

    参数:
        model (torch.nn.Module): 神经网络模型。
        trainloader (torch.utils.data.DataLoader): 训练数据集加载器。
        criterion (torch.nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        epoch (int): 当前的训练轮数。
        device (torch.device): 训练设备。

    返回:
        tuple: 包含训练损失、训练准确率和预测概率的元组。
    """
    model.train()
    running_loss = 0.0
    y_true_train = []
    y_score_train = []
    for data in tqdm(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU上

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 将模型输出转换为概率并存储真实标签和预测概率
        y_true_train.extend(labels.cpu().numpy())
        y_score_train.extend(torch.sigmoid(outputs).cpu().detach().numpy())

    train_loss = running_loss / len(trainloader)

    # 计算训练mAP
    y_true_train = torch.tensor(y_true_train)
    y_score_train = torch.tensor(y_score_train)

    train_mAP = calculate_mAP(y_true_train, y_score_train)

    # print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.3f}, Train mAP: {train_mAP:.3f}')
    return train_loss, train_mAP


def test(model, testloader, criterion, device):
    """
    测试函数，用于评估神经网络模型在测试集上的性能。

    参数:
        model (torch.nn.Module): 神经网络模型。
        testloader (torch.utils.data.DataLoader): 测试数据集加载器。
        criterion (torch.nn.Module): 损失函数。
        device (torch.device): 训练设备。

    返回:
        tuple: 包含测试损失、测试mAP和预测概率的元组。
    """
    model.eval()
    running_loss = 0.0
    y_true_test = []
    y_score_test = []
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU上

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 将模型输出转换为概率并存储真实标签和预测概率
            y_true_test.extend(labels.cpu().numpy())
            y_score_test.extend(torch.sigmoid(outputs).cpu().numpy())

    test_loss = running_loss / len(testloader)

    # 计算测试mAP
    y_true_test = torch.tensor(y_true_test)
    y_score_test = torch.tensor(y_score_test)
    test_mAP = calculate_mAP(y_true_test, y_score_test)

    # print(f'Test Loss: {test_loss:.3f}, Test mAP: {test_mAP:.3f}')

    return test_loss, test_mAP 

