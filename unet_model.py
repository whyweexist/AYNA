import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Bug/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ColorEmbedding(nn.Module):
    """Embeds color names into a learnable vector"""
    
    def __init__(self, num_colors, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_colors, embed_dim)
        
    def forward(self, color_indices):
        return self.embedding(color_indices)


class ConditionalUNet(nn.Module):
    """UNet model conditioned on color names"""
    
    def __init__(self, n_channels=1, n_classes=3, num_colors=10, embed_dim=64, bilinear=False):
        super(ConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Color embedding
        self.color_embedding = ColorEmbedding(num_colors, embed_dim)
        
        # Project color embedding to feature maps
        self.color_proj = nn.Linear(embed_dim, 64)
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, color_indices):
        # x: [B, C, H, W]
        # color_indices: [B] - indices of colors
        
        # Get color embeddings
        color_embed = self.color_embedding(color_indices)  # [B, embed_dim]
        color_features = self.color_proj(color_embed)  # [B, 64]
        
        # Expand color features to spatial dimensions
        batch_size, _, h, w = x.shape
        color_features = color_features.view(batch_size, 64, 1, 1)
        color_features = color_features.expand(batch_size, 64, h, w)
        
        # Initial convolution
        x1 = self.inc(x)
        
        # Add color conditioning to first layer
        x1 = x1 + color_features
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    # Test the model
    model = ConditionalUNet(n_channels=1, n_classes=3, num_colors=10)
    x = torch.randn(2, 1, 256, 256)
    colors = torch.randint(0, 10, (2,))
    
    with torch.no_grad():
        output = model(x, colors)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model created successfully!")