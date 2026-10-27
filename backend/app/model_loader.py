from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "inspectra_efficientnet_best.pth"
)

def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    classes = checkpoint["classes"]

    model = efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, device