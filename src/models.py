import torch.nn as nn

class DownBlock(nn.Module):
  def __init__(self, input_dim, output_dim, dropout_rate=0.5):
    super().__init__()
    self.block = nn.Sequential(
        nn.BatchNorm2d(input_dim),
        nn.Dropout2d(dropout_rate),
        nn.Conv2d(input_dim, output_dim, 3, stride=2, padding=1),
        nn.ELU()
    )
  
  def forward(self, x):
    return self.block(x)

class ConvBlock(nn.Module):
  def __init__(self, input_dim, output_dim, dropout_rate=0.5):
    super().__init__()
    self.block = nn.Sequential(
        nn.BatchNorm2d(input_dim),
        nn.Dropout2d(dropout_rate),
        nn.Conv2d(input_dim, output_dim, 5, padding='same'),
        nn.ELU()
    )
  
  def forward(self, x):
    return self.block(x)

class SegmentModel(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            DownBlock(input_dim, 32, 0.0),
            ConvBlock(32, 32),

            DownBlock(32, 64, 0.0),
            ConvBlock(64, 64),

            DownBlock(64, 128),
            ConvBlock(128, 128),
        )

        self.decoder = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 32),

            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 3, padding='same'),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
