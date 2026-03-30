import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

from SKILL_2.deployment.generator import Generator
from discriminator import Discriminator

# ================== CONFIG ==================
EPOCHS = 50
BATCH_SIZE = 64
LATENT_DIM = 100
IMAGE_SIZE = 64
CHANNELS = 1
LR = 0.0002
BETA1 = 0.5

# ----------------- DATA_PATH -----------------
DATA_PATH = "preprocessed/all_signatures"  # your preprocessed images folder
SAMPLE_DIR = "samples"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================== CUSTOM DATASET ==================
class SignatureDataset(Dataset):
    def __init__(self, root_dir, image_size=IMAGE_SIZE):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        if not self.image_paths:
            raise ValueError(f"No images found in {root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img

# ================== MAIN TRAINING ==================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ================== DATA LOADER ==================
    dataset = SignatureDataset(DATA_PATH, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    print(f"✅ Loaded {len(dataset)} images for GAN training")

    # ================== MODELS ==================
    G = Generator(latent_dim=LATENT_DIM, channels=CHANNELS).to(device)
    D = Discriminator(channels=CHANNELS).to(device)

    # ================== LOSS & OPTIMIZERS ==================
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    # ================== FIXED NOISE FOR SAMPLING ==================
    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)

    # ================== TRAINING LOOP ==================
    for epoch in range(1, EPOCHS + 1):
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ---------- TRAIN DISCRIMINATOR ----------
            optimizer_D.zero_grad()
            output_real = D(real_imgs)
            d_loss_real = criterion(output_real, real_labels)

            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_imgs = G(noise)
            output_fake = D(fake_imgs.detach())
            d_loss_fake = criterion(output_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ---------- TRAIN GENERATOR ----------
            optimizer_G.zero_grad()
            output = D(fake_imgs)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            optimizer_G.step()

            # ---------- PRINT PROGRESS ----------
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                print(f"Epoch [{epoch}/{EPOCHS}] Batch [{i+1}/{len(dataloader)}] "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # ---------- SAVE SAMPLE IMAGES EVERY EPOCH ----------
        with torch.no_grad():
            fake_samples = G(fixed_noise).detach().cpu()
            fake_samples = (fake_samples + 1) / 2  # scale to [0,1]

        utils.save_image(fake_samples, f"{SAMPLE_DIR}/epoch_{epoch:03d}.png", nrow=4, normalize=True)
        print(f"✅ Sample images saved for epoch {epoch}")

    # ================== SAVE FINAL MODELS ==================
    torch.save(G.state_dict(), f"{CHECKPOINT_DIR}/G_final.pth")
    torch.save(D.state_dict(), f"{CHECKPOINT_DIR}/D_final.pth")
    print("✅ Training complete. Models saved successfully.")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # Windows fix for multiprocessing
    main()
