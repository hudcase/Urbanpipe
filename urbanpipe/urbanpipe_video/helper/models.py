import torch
import torch.nn as nn

from transformers import TimesformerModel


class TimeSformerClassifier(nn.Module):

    def __init__(
        self,
        num_classes=17
    ):
        super().__init__()

        self.backbone = TimesformerModel.from_pretrained(
                "facebook/timesformer-base-finetuned-k400"
            )

        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):

        """
        x shape:
        (B, T, C, H, W)
        """
        # import pdb;pdb.set_trace()
        B, T, C, H, W = x.shape

        outputs = self.backbone(
            pixel_values=x
        )

        cls_token = outputs.last_hidden_state[:, 0]

        logits = self.classifier(cls_token)

        return logits


def get_model(
    model_name,
    num_classes,
    pretrained=True,
):

    if model_name == "timesformer":

        model = TimeSformerClassifier(
            num_classes=num_classes
        )

    else:
        raise ValueError("Unsupported model")

    return model
