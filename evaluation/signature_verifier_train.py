import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# ================== CONFIG ==================
# move one level UP from evaluation/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL_DIR = os.path.join(ROOT_DIR, "preprocessed", "all_signatures")
FAKE_DIR = os.path.join(ROOT_DIR, "evaluation", "generated_samples")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "evaluation", "signature_verifier.pth")

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== DATASET ==================
class SignatureDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        else:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0

        img = torch.tensor(img).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return img, label

def get_all_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(root, f))
    return images

# ================== LOAD DATA ==================
real_images = get_all_images(REAL_DIR)
fake_images = get_all_images(FAKE_DIR)

print("REAL DIR :", REAL_DIR)
print("FAKE DIR :", FAKE_DIR)
print("REAL IMAGES:", len(real_images))
print("FAKE IMAGES:", len(fake_images))

if len(real_images) == 0:
    raise ValueError("NO REAL IMAGES FOUND")
if len(fake_images) == 0:
    raise ValueError("NO FAKE IMAGES FOUND")

images = real_images + fake_images
labels = [1] * len(real_images) + [0] * len(fake_images)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

train_loader = DataLoader(
    SignatureDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    SignatureDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

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

model = SignatureVerifier().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# ================== TRAIN ==================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, lbls in train_loader:
        imgs = imgs.to(DEVICE)
        lbls = lbls.to(DEVICE).unsqueeze(1)

        preds = model(imgs)
        loss = criterion(preds, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss / len(train_loader):.4f}")

# ================== EVALUATE ==================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(DEVICE)
        lbls = lbls.to(DEVICE).unsqueeze(1)

        preds = torch.sigmoid(model(imgs))
        predicted = (preds > 0.5).float()

        correct += (predicted == lbls).sum().item()
        total += lbls.size(0)

accuracy = 100 * correct / total
print(f"\n✅ Test Accuracy: {accuracy:.2f}%")

# ================== SAVE ==================
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved at:", MODEL_SAVE_PATH)
