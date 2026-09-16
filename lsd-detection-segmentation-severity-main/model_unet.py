import torch
import torch.nn as nn
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, backbone_weights="IMAGENET1K_V1"):
        super(UNet, self).__init__()

        # Encoder: Pretrained ResNet18 backbone
        backbone = models.resnet18(weights=backbone_weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # remove avgpool & fc

        # Decoder layers - upsampling path to restore 256x256 resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 8x8 -> 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 16x16 -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 32x32 -> 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 64x64 -> 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2),  # 128x128 -> 256x256
            nn.Sigmoid()  # final activation for probability map (0–1)
        )

        # Ensure encoder weights are trainable (fine-tuning)
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out  # already between 0 and 1
