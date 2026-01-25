from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class FrameAggregatorResNet(nn.Module):
    """
    Uses a 2D ResNet backbone on individual frames and averages predictions
    over time. Input shape: (B, T, C, H, W) -> output: (B, num_classes).
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        # Newer torchvision uses weights instead of pretrained flag; handle both.
        try:
            weights = (
                models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None  # type: ignore[attr-defined]
            )
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        if x.dim() == 4:
            # Single clip without batch dimension -> (1, T, C, H, W)
            x = x.unsqueeze(0)

        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        logits = self.backbone(x)  # (B*T, num_classes)
        logits = logits.view(b, t, -1).mean(dim=1)  # average over time -> (B, num_classes)
        return logits


def load_model(model_path: str | None = None, device: str | torch.device = "cpu") -> FrameAggregatorResNet:
    model = FrameAggregatorResNet(num_classes=2, pretrained=True)
    model.to(device)
    if model_path is not None:
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
    model.eval()
    return model

