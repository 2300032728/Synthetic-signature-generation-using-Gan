import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================== PATH SETUP ==================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 🔥 CORRECT REAL IMAGE PATH (based on your screenshot)
REAL_DIR = os.path.join(
    ROOT_DIR, "signature", "sign_data", "sign_data", "train"
)

FAKE_DIR = os.path.join(
    ROOT_DIR, "evaluation", "generated_samples"
)

print("REAL_DIR :", REAL_DIR)
print("FAKE_DIR :", FAKE_DIR)
print("REAL_DIR exists:", os.path.exists(REAL_DIR))
print("FAKE_DIR exists:", os.path.exists(FAKE_DIR))


# ================== STROKE DENSITY ==================
def stroke_density(root_folder):
    densities = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(root, file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Binarize (ink = white)
            _, bin_img = cv2.threshold(
                img, 127, 255, cv2.THRESH_BINARY_INV
            )

            density = np.sum(bin_img > 0) / bin_img.size
            densities.append(density)

    return densities


# ================== RUN ==================
real_d = stroke_density(REAL_DIR)
fake_d = stroke_density(FAKE_DIR)

print(f"\nReal samples: {len(real_d)}")
print(f"Synthetic samples: {len(fake_d)}")

if len(real_d) == 0:
    print("❌ No real images found — path issue")
if len(fake_d) == 0:
    print("❌ No fake images found — generate samples first")


# ================== PLOT ==================
plt.figure(figsize=(7, 4))
plt.hist(real_d, bins=30, alpha=0.6, label="Real")
plt.hist(fake_d, bins=30, alpha=0.6, label="Synthetic")
plt.xlabel("Stroke Density")
plt.ylabel("Frequency")
plt.legend()
plt.title("Stroke Density Distribution (Real vs Synthetic)")
plt.tight_layout()
plt.show()
