import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def modify_fc_layer(model, num_classes):
    # Vision Transformer (ViT)
    if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    else:
        raise ValueError("The model does not contain a compatible final layer.")

    return model


def get_model(model_name="vit_h_14", num_classes=17):
    print('==> model name {}'.format(model_name))

    # =========================
    # Vision Transformer (ViT)
    # =========================
    # vit 预训练的torch中，只有这个型号的支持大分辨率输入：https://docs.pytorch.org/vision/stable/models/vision_transformer.html
    if model_name == "vit_h_14":
        model = models.vit_h_14(weights='DEFAULT')
        model = modify_fc_layer(model, num_classes)
    else:
        raise ValueError("Unknown model value of : {}".format(model_name))

    return model


# =========================
# 测试代码
# =========================

if __name__ == "__main__":

    model = get_model(
        model_name="vit_h_14",
        num_classes=17,
        pretrained=True
    )

    print(model)

    # 测试 forward
    x = torch.randn(1, 3, 518, 518)
    y = model(x)

    print("output shape:", y.shape)
