import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np

# ================== CONFIG ==================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL_DIR = os.path.join(ROOT_DIR, "preprocessed", "all_signatures")
FAKE_DIR = os.path.join(ROOT_DIR, "evaluation", "generated_samples")
MODEL_PATH = os.path.join(ROOT_DIR, "evaluation", "signature_verifier.pth")
OUTPUT_CSV = os.path.join(ROOT_DIR, "evaluation", "verifier_predictions.csv")

IMG_SIZE = 64
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== DATASET ==================
class SignatureDataset(Dataset):
    def __init__(self, img_dir, label):
        self.images = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(root, f))
        self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        else:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0

        img = torch.tensor(img).unsqueeze(0)
        label = torch.tensor(self.label, dtype=torch.float32)

        return img, label, path

# ================== MODEL ==================
class SignatureVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# ================== LOAD DATA ==================
real_ds = SignatureDataset(REAL_DIR, 1)
fake_ds = SignatureDataset(FAKE_DIR, 0)
dataset = ConcatDataset([real_ds, fake_ds])

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================== LOAD MODEL ==================
model = SignatureVerifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ================== EVALUATE ==================
preds, labels, paths = [], [], []

with torch.no_grad():
    for imgs, lbls, pths in loader:
        imgs = imgs.to(DEVICE)
        outputs = torch.sigmoid(model(imgs))
        pred = (outputs > 0.5).float().cpu().numpy()

        preds.extend(pred.flatten().tolist())
        labels.extend(lbls.numpy().tolist())
        paths.extend(pths)

accuracy = sum(p == l for p, l in zip(preds, labels)) / len(preds)

print("\n====== VERIFIER EVALUATION ======")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Samples  : {len(preds)}")
print(f"Real     : {sum(labels)}")
print(f"Fake     : {len(labels) - sum(labels)}")

# ================== SAVE CSV ==================
df = pd.DataFrame({
    "image_path": paths,
    "label": labels,
    "prediction": preds
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n📁 Predictions saved to {OUTPUT_CSV}")
