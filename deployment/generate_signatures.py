import torch
import os
import cv2
import numpy as np
from generator import Generator   # ✅ CORRECT ONE


def generate_signatures(N=5, output_dir="generated", model_path="../checkpoints/G_final.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    G = Generator(latent_dim=100, channels=1).to(device)  # ✅ MATCH TRAIN.PY

    checkpoint = torch.load(model_path, map_location=device)
    G.load_state_dict(checkpoint)

    G.eval()

    image_paths = []

    for i in range(N):
        z = torch.randn(1, 100, 1, 1, device=device)

        with torch.no_grad():
            fake = G(z).cpu().numpy()[0][0]

        fake = (fake + 1) / 2
        fake = (fake * 255).astype(np.uint8)

        img_path = os.path.join(output_dir, f"signature_{i}.png")
        cv2.imwrite(img_path, fake)

        image_paths.append(img_path)

    return image_paths