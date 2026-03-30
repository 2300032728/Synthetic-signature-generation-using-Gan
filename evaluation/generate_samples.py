# generate_samples.py
import os
import sys
import torch
from torchvision.utils import save_image

# ================== FIX IMPORT PATH ==================
# Add SKILL_2 root directory to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from generator import Generator  # NOW THIS WORKS

# ================== CONFIG ==================
LATENT_DIM = 100
CHANNELS = 1
NUM_SAMPLES = 16
OUTPUT_DIR = "generated_samples"
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints", "G_final.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================== LOAD GENERATOR ==================
generator = Generator(latent_dim=LATENT_DIM, channels=CHANNELS).to(device)
generator.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
generator.eval()
print("✅ Generator loaded successfully")

# ================== GENERATE SAMPLES ==================
with torch.no_grad():
    z = torch.randn(NUM_SAMPLES, LATENT_DIM, 1, 1, device=device)
    fake_images = generator(z)
    fake_images = (fake_images + 1) / 2  # [-1,1] → [0,1]

# ================== SAVE IMAGES ==================
for i, img in enumerate(fake_images):
    save_image(img, os.path.join(OUTPUT_DIR, f"sample_{i+1}.png"))

print(f"✅ {NUM_SAMPLES} images saved in '{OUTPUT_DIR}'")
