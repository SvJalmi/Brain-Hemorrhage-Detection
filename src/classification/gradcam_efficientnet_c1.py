import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models

from src.classification.transforms import eval_transform


# ============================================================
# C1 — GRAD-CAM EXPLAINABILITY
# EfficientNet-B0 Frozen Baseline
# ============================================================

CHECKPOINT_PATH = (
    "models/classification/"
    "best_efficientnet_b0_frozen_tbihemoct.pth"
)

OUTPUT_DIR = Path("results/classification/gradcam")

NUM_CLASSES = 5

CLASS_NAMES = [
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
]


print("=" * 70)
print("C1-FROZEN — GRAD-CAM EXPLAINABILITY")
print("EfficientNet-B0")
print("=" * 70)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nDevice: {device}")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(weights=None)

model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES,
)

model = model.to(device)

print("\nLoading C1 checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])

print(f"Checkpoint epoch: {checkpoint['epoch']}")

model.eval()


# ------------------------------------------------------------
# Grad-CAM target layer
# ------------------------------------------------------------

target_layer = model.features[-1]

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ------------------------------------------------------------
# Select an image
# ------------------------------------------------------------

print("\nEnter the path of a CT image.")
print("Example:")
print(
    "datasets/TBI-HemoCT/"
    "TBI-HemoCT Data set/TBI-HemoCT Data set/"
    "Subdural/ID_1_subdural_UNIQUE.png"
)

image_path = input("\nImage path: ").strip().strip('"')


if not Path(image_path).exists():
    print("\nERROR: Image path does not exist.")
    forward_handle.remove()
    backward_handle.remove()
    sys.exit(1)


# ------------------------------------------------------------
# Load image
# ------------------------------------------------------------

print("\nLoading image...")

original_image = Image.open(image_path).convert("L")

input_tensor = eval_transform(
    original_image.convert("RGB")
).unsqueeze(0).to(device)


# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------

print("Running model inference...")

model.zero_grad()

output = model(input_tensor)

probabilities = torch.sigmoid(output)[0]

predicted_class = torch.argmax(probabilities).item()

predicted_probability = probabilities[predicted_class].item()


print("\nModel predictions:")

for class_idx, class_name in enumerate(CLASS_NAMES):

    print(
        f"  {class_name:18s}: "
        f"{probabilities[class_idx].item():.4f}"
    )


print(
    f"\nHighest-probability class: "
    f"{CLASS_NAMES[predicted_class]}"
)

print(
    f"Probability: {predicted_probability:.4f}"
)


# ------------------------------------------------------------
# Backward pass for predicted class
# ------------------------------------------------------------

print("\nGenerating Grad-CAM...")

score = output[0, predicted_class]

score.backward()


# ------------------------------------------------------------
# Compute Grad-CAM
# ------------------------------------------------------------

weights = gradients.mean(
    dim=(2, 3),
    keepdim=True,
)

cam = (weights * activations).sum(
    dim=1,
)

cam = torch.relu(cam)

cam = cam.squeeze().detach().cpu().numpy()


# Normalize CAM

cam -= cam.min()

if cam.max() > 0:
    cam /= cam.max()


# Resize CAM to original image size

original_width, original_height = original_image.size

cam_image = Image.fromarray(
    np.uint8(cam * 255)
).resize(
    (original_width, original_height),
    Image.Resampling.BILINEAR,
)

cam = np.asarray(cam_image) / 255.0


# ------------------------------------------------------------
# Save Grad-CAM visualization
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

safe_class_name = CLASS_NAMES[predicted_class]

output_path = (
    OUTPUT_DIR /
    f"gradcam_{safe_class_name}.png"
)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)

plt.imshow(
    original_image,
    cmap="gray"
)

plt.title("Original CT")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    original_image,
    cmap="gray"
)

plt.imshow(
    cam,
    cmap="jet",
    alpha=0.45
)

plt.title(
    f"Grad-CAM\n{safe_class_name}"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    cam,
    cmap="jet"
)

plt.title("Activation Map")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------

forward_handle.remove()
backward_handle.remove()


print("\n" + "=" * 70)
print("GRAD-CAM COMPLETE")
print("=" * 70)

print(f"\nSaved visualization:")
print(output_path)

print(
    "\nInterpretation:"
)

print(
    "The highlighted regions indicate areas that contributed "
    "most strongly to the model's prediction for the selected class."
)

print("=" * 70)