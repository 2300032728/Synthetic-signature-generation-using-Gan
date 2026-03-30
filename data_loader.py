import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# -------------------------------
# Configuration
# -------------------------------
RAW_DATA_DIR = "signature/sign_data"
PREPROCESSED_DIR = "preprocessed/all_signatures"
IMAGE_SIZE = 256          # ✅ 256x256
BATCH_SIZE = 64
MAX_IMAGES = 10000

# -------------------------------
# Step 1: Preprocess & save images
# -------------------------------
def preprocess_and_save_images(
    raw_dir=RAW_DATA_DIR,
    save_dir=PREPROCESSED_DIR,
    image_size=IMAGE_SIZE,
    max_images=MAX_IMAGES
):
    os.makedirs(save_dir, exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = []

    # Collect all images from raw_dir
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        raise ValueError(f"❌ No images found in {raw_dir}")

    print(f"✅ Found {len(image_paths)} raw images.")
    print(f"🔄 Resizing all images to {image_size}x{image_size}...")

    count = 0
    for path in image_paths:
        if count >= max_images:
            break

        try:
            img = Image.open(path).convert("L")  # Convert to grayscale
            img = img.resize((image_size, image_size), Image.LANCZOS)

            save_path = os.path.join(save_dir, f"signature_{count}.png")
            img.save(save_path)

            count += 1
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")

    print(f"✅ Saved {count} resized images to {save_dir}")

# -------------------------------
# Step 2: PyTorch Dataset
# -------------------------------
class SignatureDataset(Dataset):
    def __init__(self, root_dir, image_size=IMAGE_SIZE):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        if not self.image_paths:
            raise ValueError(f"❌ No images found in {root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image

# -------------------------------
# Step 3: DataLoader helper
# -------------------------------
def get_dataloader(
    root_dir=PREPROCESSED_DIR,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True
):
    dataset = SignatureDataset(root_dir, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        drop_last=True
    )

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":

    # 1️⃣ Preprocess raw images → 256x256
    preprocess_and_save_images()

    # 2️⃣ Load preprocessed images
    dataloader = get_dataloader()

    print(f"✅ Loaded {len(dataloader.dataset)} images for GAN training")

    # 3️⃣ Test one batch
    for batch in dataloader:
        print("Batch shape:", batch.shape)  # (64, 1, 256, 256)
        print("Min pixel value:", batch.min().item())
        print("Max pixel value:", batch.max().item())
        break