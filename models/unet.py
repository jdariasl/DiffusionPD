import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, class_embeb_dim, time_emb_dim):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.class_embeb = nn.Linear(class_embeb_dim, out_channels)

    def forward(self, x, class_embeb, time_emb):
        time_emb = self.time_emb(time_emb).unsqueeze(2).repeat(1, 1, x.size(2))
        class_embeb = self.class_embeb(class_embeb).unsqueeze(2).repeat(1, 1, x.size(2))
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.silu(x1) + time_emb + class_embeb
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = F.silu(x1)
        return x1 + x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_classes=4,
        init_features=32,
        time_emb_dim=128,
    ):
        super(UNet, self).__init__()

        self.num_classes = num_classes
        self.time_emb_dim = time_emb_dim
        features = init_features
        self.encoder1 = UNetBlock(in_channels, features, init_features, time_emb_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNetBlock(features, features * 2, init_features, time_emb_dim)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNetBlock(
            features * 2, features * 4, init_features, time_emb_dim
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNetBlock(
            features * 4, features * 8, init_features, time_emb_dim
        )
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck_1 = UNetBlock(
            features * 8, features * 8, init_features, time_emb_dim
        )
        self.bottleneck_2 = UNetBlock(
            features * 8, features * 16, init_features, time_emb_dim
        )

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNetBlock(
            features * 16, features * 8, init_features, time_emb_dim
        )
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNetBlock(
            features * 8, features * 4, init_features, time_emb_dim
        )
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNetBlock(
            features * 4, features * 2, init_features, time_emb_dim
        )
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNetBlock(features * 2, features, init_features, time_emb_dim)

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        # Class embedding
        self.class_mlp = nn.Sequential(nn.Linear(num_classes, init_features), nn.SiLU())

    def encode(self, x, class_label, time_step):
        # One-hot encode the class label and expand dimensions to match input
        class_onehot = F.one_hot(class_label, num_classes=self.num_classes).float()
        class_onehot = class_onehot.view(class_onehot.size(0), self.num_classes)
        class_embeb = self.class_mlp(class_onehot)

        x = x.view(x.size(0), 1, x.size(1))
        # Concatenate the class conditional input with the input tensor
        # x = torch.cat((x, class_embeb.view(class_embeb.size(0), 1, class_embeb.size(1))), dim=1)
        time_emb = self.time_mlp(time_step)

        enc1 = self.encoder1(x, class_embeb, time_emb)
        enc2 = self.encoder2(self.pool1(enc1), class_embeb, time_emb)
        enc3 = self.encoder3(self.pool2(enc2), class_embeb, time_emb)
        enc4 = self.encoder4(self.pool3(enc3), class_embeb, time_emb)

        bottleneck = self.bottleneck_1(self.pool4(enc4), class_embeb, time_emb)
        bottleneck = self.bottleneck_2(bottleneck, class_embeb, time_emb)

        return bottleneck, enc1, enc2, enc3, enc4, class_embeb, time_emb

    def forward(self, x, class_label, time_step):

        bottleneck, enc1, enc2, enc3, enc4, class_embeb, time_emb = self.encode(
            x, class_label, time_step
        )

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4, class_embeb, time_emb)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3, class_embeb, time_emb)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2, class_embeb, time_emb)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1, class_embeb, time_emb)
        out = torch.sigmoid(self.conv(dec1))
        return out.squeeze(1)
