import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    一个简单的UNet网络
    """

    def __init__(self, time_emb_dim=32):
        super().__init__()

        # 对正余弦的编码结果，再使用mlp进行一次编码，让编码过程是可学习的
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2), nn.ReLU(), nn.Linear(time_emb_dim * 2, time_emb_dim * 2)
        )
        # 时间编码的结果向高维映射，方便和卷积层拼接
        self.time_emb_proj = nn.Linear(time_emb_dim * 2, 256)

        # 三个用于编码的卷积层
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        self.bottleneck = self.conv_block(256, 512)

        # 三个用于解码的卷积层
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)

        # 池化用于下采样
        self.pool = nn.MaxPool2d(2)
        # bilinear的方式上采样
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # 最终输出的卷积层
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        一个简单的卷积Block,不改变图片的长宽, 只改变图片的通道数
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def get_time_embedding(self, timestep, dim=32, max_period=10000):
        """
        正余弦时间编码, 这里参考Transformer的 正余弦位置编码, 道理都是一样的
        """
        half = dim // 2
        freqs = torch.exp(-torch.arange(half, dtype=torch.float32) * torch.log(torch.tensor(max_period)) / half).to(
            timestep.device
        )
        args = timestep.unsqueeze(-1).float() * freqs.unsqueeze(0)  # (B, half)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return embedding

    def forward(self, x, t):
        batch_size = x.shape[0]
        # 先进行时间映射
        t_emb = self.get_time_embedding(t).to(x.device)  # (batch_size, time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # (batch_size, time_emb_dim*2)
        time_emb = self.time_emb_proj(t_emb)  # (B, 256)
        time_emb = time_emb.view(
            batch_size, -1, 1, 1
        )  # (B, 256, 1, 1)  变换成和中间层的卷积同样的维度，这样通过广播加法可以很容易的apply到卷积层里
        # 编码过程 (编码 -> 下采样) -> (编码 -> 下采样) -> (编码 -> 下采样)
        e1 = self.enc1(x)  # (B, 64, 32, 32)
        e2 = self.enc2(self.pool(e1))  # (B, 128, 16, 16)
        e3 = self.enc3(self.pool(e2))  # (B, 256, 8, 8)

        # 在bottleneck之前，apply时间编码
        e3 = e3 + time_emb  # 广播加法
        b = self.bottleneck(self.pool(e3))  # (B, 512, 4, 4)

        # 解码过程 (上采样 -> 拼接 -> 解码) -> (上采样 -> 拼接 -> 解码) -> (上采样 -> 拼接 -> 解码)
        d3 = self.upsample(b)  # (B, 512, 8, 8)
        d3 = torch.cat([d3, e3], dim=1)  # (B, 512+256, 8, 8)
        d3 = self.dec3(d3)  # (B, 256, 8, 8)
        d2 = self.upsample(d3)  # (B, 256, 16, 16)
        d2 = torch.cat([d2, e2], dim=1)  # (B, 256+128, 16, 16)
        d2 = self.dec2(d2)  # (B, 128, 16, 16)
        d1 = self.upsample(d2)  # (B, 128, 32, 32)
        d1 = torch.cat([d1, e1], dim=1)  # (B, 128+64, 32, 32)
        d1 = self.dec1(d1)  # (B, 64, 32, 32)

        # 最终输出
        out = self.final_conv(d1)  # (B, 3, 32, 32)
        # out = torch.sigmoid(out)
        return out


#
#
#
