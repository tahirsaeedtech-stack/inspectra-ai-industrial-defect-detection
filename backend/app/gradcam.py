from .preprocessing import preprocess_image
from PIL import Image
from matplotlib import colormaps
import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")


def generate_gradcam(image, model, classes, device):
    """
    Generate a Grad-CAM visualization for the predicted class
    using the EfficientNet-B0 model.
    """

    # Step 1: Preprocess input image
    input_tensor = preprocess_image(image).to(device)

    activations = {}
    gradients = {}

    # Step 2: Select final convolutional feature stage
    target_layer = model.features[-1]

    def forward_hook(module, inputs, output):
        activations["value"] = output.detach()

        def save_gradient(gradient):
            gradients["value"] = gradient.detach()

        output.register_hook(save_gradient)

    # Register hook
    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.eval()

        # Step 3: Forward and backward passes
        with torch.enable_grad():

            model.zero_grad(set_to_none=True)

            output = model(input_tensor)

            predicted_idx = output.argmax(dim=1).item()

            score = output[0, predicted_idx]

            score.backward()

        # Step 4: Retrieve activations and gradients
        feature_maps = activations["value"]
        gradient_maps = gradients["value"]

        # Step 5: Calculate Grad-CAM weights
        weights = gradient_maps.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (weights * feature_maps).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        # Step 6: Resize heatmap
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].cpu().numpy()

        # Step 7: Normalize heatmap
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Step 8: Generate colorful JET heatmap
        # Blue = low activation
        # Green/yellow = moderate activation
        # Red = high activation

        colored_cam = colormaps["jet"](cam)[:, :, :3]

        heatmap = (colored_cam * 255).astype(np.uint8)

        heatmap_image = Image.fromarray(heatmap)

        # Step 9: Prepare original image
        original = image.convert("RGB").resize((224, 224))

        # Step 10: Overlay heatmap on original image
        overlay = Image.blend(
            original,
            heatmap_image,
            alpha=0.45
        )

        # Step 11: Convert overlay to Base64
        buffer = BytesIO()

        overlay.save(buffer, format="PNG")

        encoded_image = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        # Step 12: Return API response
        return {
            "prediction": classes[predicted_idx],
            "gradcam_image": (
                f"data:image/png;base64,{encoded_image}"
            )
        }

    finally:
        # Remove hook and clear gradients
        handle.remove()
        model.zero_grad(set_to_none=True)
