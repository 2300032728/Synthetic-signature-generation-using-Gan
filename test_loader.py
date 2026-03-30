from data_loader import load_data
import matplotlib.pyplot as plt

dataset = load_data(
    image_size=128,
    batch_size=8,
    shuffle=True
)

for batch in dataset.take(1):
    images = batch.numpy()

    plt.figure(figsize=(8, 8))
    for i in range(8):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.axis("off")

    plt.suptitle("✅ Preprocessed Images")
    plt.show()
