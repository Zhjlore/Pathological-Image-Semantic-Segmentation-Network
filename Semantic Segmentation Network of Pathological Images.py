import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.down(x)


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dilated_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=6, dilation=6)
        self.dilated_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=12, dilation=12)
        self.dilated_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=18, dilation=18)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fuse = DoubleConv(in_channels * 4, in_channels)

    def forward(self, x):
        x1 = self.dilated_conv1(x)
        x2 = self.dilated_conv2(x)
        x3 = self.dilated_conv3(x)
        x_global = self.global_avg_pool(x)
        x_global = x_global.expand(x.size())
        fused = torch.cat((x1, x2, x3, x_global), dim=1)
        return self.fuse(fused)


class PositionAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention).view(batch_size, channels, height, width)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query(x).view(batch_size, -1, height * width)
        proj_key = self.key(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, -1, height, width)
        out = self.out_conv(out)
        return out


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.position_attention = PositionAttention(in_channels)
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, x):
        position_out = self.position_attention(x)
        channel_out = self.channel_attention(x)
        return position_out + channel_out


class UpSample(nn.Module):
    def __init__(self, x1_channels, x2_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(x1_channels, out_channels, kernel_size=2, stride=2)
        self.conv_layer = DoubleConv(out_channels + x2_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv_layer(x)


class FinalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DownSample(3, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        self.attention = AttentionModule(512)
        self.fusion = MultiScaleFusion(512)

        self.up1 = UpSample(512, 256, 256)
        self.up2 = UpSample(256, 128, 128)
        self.up3 = UpSample(128, 64, 64)
        self.up4 = UpSample(64, 64, 32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x_attention = self.attention(x4)
        x_fused = self.fusion(x_attention)

        x_up1 = self.up1(x_fused, x3)
        x_up2 = self.up2(x_up1, x2)
        x_up3 = self.up3(x_up2, x1)
        x_up4 = self.up4(x_up3, x1)  

        output = self.final_conv(x_up4)
        return output


# 测试网络结构
if __name__ == "__main__":
    model = FinalNetwork()
    input_tensor = torch.randn(1, 3, 512, 512)
    output = model(input_tensor)
    print("Output shape:", output.shape)
