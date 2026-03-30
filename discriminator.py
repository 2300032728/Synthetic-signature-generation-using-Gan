import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels=1, feature_map=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, feature_map, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map, feature_map*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map*2, feature_map*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map*4, feature_map*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)