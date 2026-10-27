import time
import torch
from .preprocessing import preprocess_image


def predict_image(image, model, classes, device):

    input_tensor = preprocess_image(image).to(device)

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) * 1000

    confidence, predicted_idx = torch.max(
        probabilities, dim=0
    )

    return {
        "prediction": classes[predicted_idx.item()],
        "confidence": round(confidence.item(), 4),
        "processing_time_ms": round(elapsed, 2),
        "model": "efficientnet_b0",
        "probabilities": {
            class_name: round(probabilities[i].item(), 4)
            for i, class_name in enumerate(classes)
        }
    }
