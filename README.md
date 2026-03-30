Synthetic Signature Generation using GAN
A deep learning project that uses Generative Adversarial Networks (GANs) to synthesize realistic handwritten signatures. The generated signatures can be used for data augmentation, signature verification research, and biometric security testing.

Table of Contents

Overview
Project Structure
Architecture
Dataset
Installation
Usage
Results
Evaluation Metrics
Sample Outputs
Future Work
References
License


Overview
Handwritten signatures are a widely used biometric identifier. Collecting real signature data is expensive and raises privacy concerns. This project tackles that challenge by training a GAN to generate synthetic signatures that are:

Visually realistic and indistinguishable from genuine signatures
Diverse across different writer styles
Useful for augmenting small real-world datasets

The model is trained on the CEDAR and GPDS Synthetic Signature datasets and evaluated using FID score, SSIM, and a downstream signature verification task.

Project Structure
synthetic-signature-gan/
│
├── data/
│   ├── raw/                    # Original dataset (not tracked by git)
│   ├── processed/              # Preprocessed signature images
│   └── augmented/              # GAN-generated synthetic signatures
│
├── models/
│   ├── generator.py            # Generator network definition
│   ├── discriminator.py        # Discriminator network definition
│   └── checkpoints/            # Saved model weights (.pth files)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
│
├── src/
│   ├── dataset.py              # Custom PyTorch dataset class
│   ├── train.py                # Training loop
│   ├── generate.py             # Generate synthetic signatures
│   ├── evaluate.py             # FID, SSIM evaluation
│   └── utils.py                # Helper functions
│
├── outputs/
│   ├── generated_samples/      # Output images from generator
│   └── training_plots/         # Loss curves, metric plots
│
├── app/
│   └── streamlit_app.py        # Interactive demo UI
│
├── requirements.txt
├── config.yaml                 # Hyperparameters and settings
├── .gitignore
└── README.md

Architecture
The project uses a DCGAN (Deep Convolutional GAN) with the following design:
Generator

Input: Random noise vector z of size 100 (latent space)
Layers: Transposed Convolution → BatchNorm → ReLU (×4)
Output: 64×64 grayscale signature image
Final activation: Tanh

Discriminator

Input: 64×64 grayscale image (real or generated)
Layers: Convolution → BatchNorm → LeakyReLU (×4)
Output: Scalar probability (real vs. fake)
Final activation: Sigmoid

Loss Function
Binary Cross-Entropy (BCE) loss with the standard GAN minimax objective:
min_G max_D [ E[log D(x)] + E[log(1 - D(G(z)))] ]

Dataset
DatasetSignaturesWritersTypeCEDAR2,64055Genuine + ForgedGPDS-Synthetic24,000600GenuineGenerated (ours)10,000—Synthetic
Preprocessing steps:

Resize all images to 64×64 pixels
Convert to grayscale
Normalize pixel values to [-1, 1]
Apply light augmentation (rotation ±5°, slight zoom)


Raw dataset files are not included in this repository. See Dataset Download below.

Dataset Download

CEDAR: Available here
GPDS: Request access at the official GPDS website

Place downloaded data in data/raw/ and run:
bashpython src/dataset.py --prepare

Installation
Prerequisites

Python 3.8+
CUDA-compatible GPU (recommended: 4GB+ VRAM)
Git

Steps
bash# 1. Clone the repository
git clone https://github.com/your-username/synthetic-signature-gan.git
cd synthetic-signature-gan

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
requirements.txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-image>=0.20.0
scipy>=1.10.0
Pillow>=9.4.0
tqdm>=4.65.0
streamlit>=1.22.0
pyyaml>=6.0
tensorboard>=2.12.0

Usage
1. Train the model
bashpython src/train.py --config config.yaml
Key parameters in config.yaml:
yamltraining:
  epochs: 200
  batch_size: 64
  learning_rate_g: 0.0002
  learning_rate_d: 0.0002
  latent_dim: 100
  beta1: 0.5

model:
  image_size: 64
  channels: 1
  features_g: 64
  features_d: 64

data:
  dataset_path: data/processed/
  train_split: 0.85
Training logs and checkpoints are saved every 10 epochs to models/checkpoints/.
Monitor training with TensorBoard:
bashtensorboard --logdir outputs/training_plots/
2. Generate synthetic signatures
bash# Generate 100 signatures
python src/generate.py --checkpoint models/checkpoints/gen_epoch_200.pth --count 100 --output outputs/generated_samples/
3. Evaluate the model
bashpython src/evaluate.py --real data/processed/ --fake outputs/generated_samples/
4. Run the Streamlit demo
bashstreamlit run app/streamlit_app.py
Open your browser at http://localhost:8501

Results
The model was trained for 200 epochs on an NVIDIA RTX 3060 (6GB VRAM), taking approximately 3.5 hours.
Training Curves
EpochGenerator LossDiscriminator Loss104.210.38502.870.481001.940.521501.630.552001.410.57
Generator loss decreasing while discriminator stabilizes near 0.5 indicates healthy GAN training.

Evaluation Metrics
MetricValueDescriptionFID Score18.4Lower is better. < 20 is considered high qualitySSIM0.76Structural similarity to real signatures (0–1)IS (Inception Score)3.2Higher is betterVerification EER8.3%Equal Error Rate on downstream verification task

FID (Fréchet Inception Distance) measures the statistical distance between real and generated image distributions. Our score of 18.4 is competitive with published results on signature generation tasks.


Sample Outputs
Generated signatures after 200 epochs of training:
outputs/generated_samples/
├── sample_001.png   ← clean cursive style
├── sample_002.png   ← looped signature
├── sample_003.png   ← angular short signature
└── ...
Visual quality improves significantly after epoch 80, with stable, diverse outputs from epoch 150 onward.


