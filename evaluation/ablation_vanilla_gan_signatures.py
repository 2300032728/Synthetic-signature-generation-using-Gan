import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

# ================= CONFIG =================
LATENT_DIMS = [50, 100, 200]
TRAINED_Z = 100
NUM_SAMPLES = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "checkpoints", "G_final.pth")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= EXACT TRAINED GENERATOR =================
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# ================= LOAD GENERATOR =================
print("🔄 Loading Generator (z=100)...")
generator = Generator(z_dim=TRAINED_Z).to(DEVICE)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
generator.eval()
print("✅ Generator loaded correctly")

# ================= ABLATION =================
for z_dim in LATENT_DIMS:
    print(f"\n🔬 Ablation for latent dim = {z_dim}")

    z_full = torch.zeros(NUM_SAMPLES, TRAINED_Z, 1, 1, device=DEVICE)

    if z_dim <= TRAINED_Z:
        z_partial = torch.randn(NUM_SAMPLES, z_dim, 1, 1, device=DEVICE)
        z_full[:, :z_dim] = z_partial
    else:
        z_full = torch.randn(NUM_SAMPLES, TRAINED_Z, 1, 1, device=DEVICE)

    with torch.no_grad():
        fake_images = generator(z_full)

    save_path = os.path.join(OUTPUT_DIR, f"generated_z{z_dim}.png")
    save_image(fake_images, save_path, nrow=4, normalize=True)

    print(f"✅ Saved → {save_path}")

print("\n🎯 Ablation completed successfully.")
