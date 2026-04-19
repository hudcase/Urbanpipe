import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def modify_fc_layer(model, num_classes):
    """
    修改模型的最后分类层，使输出类别数量为 num_classes。

    支持：
    - ResNet
    - ViT
    - AlexNet
    - VGG
    """

    # ResNet
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Vision Transformer (ViT)
    elif hasattr(model, 'heads') and hasattr(model.heads, 'head'):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    else:
        raise ValueError("The model does not contain a compatible final layer.")

    return model


def get_model(model_name="resnet18", num_classes=16, pretrained=False):
    """
    获取指定模型实例。

    参数:
        model_name (str): 模型名称
        num_classes (int): 类别数量
        pretrained (bool): 是否使用预训练权重

    返回:
        torch.nn.Module
    """

    print('==> model name {}'.format(model_name))

    # =========================
    # ResNet 系列
    # =========================

    if model_name == "resnet18":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.resnet18(weights=weights)
        model = modify_fc_layer(model, num_classes)

    elif model_name == "resnet34":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.resnet34(weights=weights)
        model = modify_fc_layer(model, num_classes)

    elif model_name == "resnet50":
        weights = "IMAGENET1K_V2" if pretrained else None
        model = models.resnet50(weights=weights)
        model = modify_fc_layer(model, num_classes)

    elif model_name == "resnet101":
        weights = "IMAGENET1K_V2" if pretrained else None
        model = models.resnet101(weights=weights)
        model = modify_fc_layer(model, num_classes)

    elif model_name == "resnet152":
        weights = "IMAGENET1K_V2" if pretrained else None
        model = models.resnet152(weights=weights)
        model = modify_fc_layer(model, num_classes)


    # =========================
    # Vision Transformer (ViT)
    # =========================

    elif model_name == "vit_b_16":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.vit_b_16(weights=weights)
        model = modify_fc_layer(model, num_classes)

    elif model_name == "vit_b_32":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.vit_b_32(weights=weights)
        model = modify_fc_layer(model, num_classes)

    elif model_name == "vit_l_16":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.vit_l_16(weights=weights)
        model = modify_fc_layer(model, num_classes)

    else:
        raise ValueError("Unknown model value of : {}".format(model_name))

    return model


# =========================
# 测试代码
# =========================

if __name__ == "__main__":

    model = get_model(
        model_name="vit_b_16",
        num_classes=16,
        pretrained=True
    )

    print(model)

    # 测试 forward
    x = torch.randn(1, 3, 224, 224)
    y = model(x)

    print("output shape:", y.shape)
