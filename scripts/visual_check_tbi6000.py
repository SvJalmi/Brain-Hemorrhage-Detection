from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

ROOT = Path(r"datasets\TBI-6000")
OUTPUT = Path(r"dataset_check\tbi6000_visual_samples.png")

classes = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "normal",
    "subarachnoid",
    "subdural",
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# ---------------------------------------------------------
# First six panels: one random image from each class
# ---------------------------------------------------------
for i, class_name in enumerate(classes):
    class_dir = ROOT / class_name
    images = list(class_dir.glob("*.png"))

    selected = random.choice(images)

    img = Image.open(selected).convert("L")

    axes.flat[i].imshow(img, cmap="gray")
    axes.flat[i].set_title(
        f"{class_name}\n{selected.name}\n{img.size}"
    )
    axes.flat[i].axis("off")

# ---------------------------------------------------------
# Seventh panel: unusual 512x666 image
# ---------------------------------------------------------
unusual = ROOT / "subdural" / "ID_315_subdural_UNIQUE.png"

img = Image.open(unusual).convert("L")

axes.flat[6].imshow(img, cmap="gray")
axes.flat[6].set_title(
    f"UNUSUAL IMAGE\n{unusual.name}\n{img.size}"
)
axes.flat[6].axis("off")

# ---------------------------------------------------------
# Eighth panel: same image resized ONLY for display
# ---------------------------------------------------------
axes.flat[7].imshow(img.resize((512, 512)), cmap="gray")
axes.flat[7].set_title(
    "UNUSUAL IMAGE\nDisplay resized to 512x512"
)
axes.flat[7].axis("off")

plt.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=150)
plt.close()

print("Visual inspection image created:")
print(OUTPUT)