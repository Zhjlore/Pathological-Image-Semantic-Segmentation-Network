import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    双卷积层 结合BN和GN
    3x3 卷积+批归一化+ReLU+3x3 卷积+组归一化+ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_channels),  
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    """
    下采样模块
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self,x):
        return self.down(x)


class MultiScaleFusion(nn.Module):
    """
    多尺度特征提取与融合模块
    """
    def __init__(self,in_channels):
        super().__init__()
        self.dilated_conv1 = nn.Conv2d(in_channels, in_channels,kernel_size=3, padding=6,dilation=6)
        self.dilated_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=12, dilation=12)
        self.dilated_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=18, dilation=18)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fuse = DoubleConv(in_channels*4, in_channels)

    def forward(self,x):
        x1 = self.dilated_conv1(x)
        x2 = self.dilated_conv2(x)
        x3 = self.dilated_conv3(x)
        x_global = self.global_avg_pool(x)
        # 扩展到与 x 相同的尺寸
        x_global = x_global.expand(x.size())
        fused = torch.cat((x1, x2, x3, x_global), dim=1)
        return self.fuse(fused)


class PositionAttention(nn.Module):
    """
    位置注意力模块
    """
    def __init__(self,in_channels):
        super().__init__()
        # 查询、键和值的计算，通过1*1卷积实现，分别降维或保持通道数不变
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  

    def forward(self,x):
        batch_size,channels,height,width = x.size()
        query = self.query(x).view(batch_size, -1, height*width).permute(0,2,1)  # (B, H*W, C/8)
        key = self.key(x).view(batch_size, -1, height*width)   # (B, C/8, H*W)
        value = self.value(x).view(batch_size, -1, height * width)  # (batch_size, in_channels, height * width)
        # 计算关联强度矩阵
        # input（p,m,n) * mat2(p,n,a) ->output(p,m,a)
        attention = self.softmax(torch.bmm(query,key)) # (batch_size, height * width, height * width)
        # 加权融合
        out = torch.bmm(value, attention) # (B, C, H*W)
        out = out.view(batch_size, channels, height, width)
        return out


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height*width) # (B, C/8, H*W)
        key = self.key(x).view(batch_size, -1, height*width).permute(0,2,1) # (B, H*W, C/8)
        value = self.value(x).view(batch_size, -1, height*width) # (B, C/8, H*W)
        # 计算注意力分数：每个通道与其他通道的相似性
        attention = self.softmax(torch.bmm(query, key)) # (B, C/8, C/8)
        # 使用注意力分数加权特征
        out = torch.bmm(value, attention)  # (B, C, H*W)
        out = out.view(batch_size, channels, height, width)
        return out


class AttentionModule(nn.Module):
    """
    注意力模块，融合位置注意力和通道注意力
    """
    def __init__(self, in_channels):
        super().__init__()
        self.position_attention = PositionAttention(in_channels)
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self,x):
        position_out = self.position_attention(x)
        channel_out = self.channel_attention(x)
        return position_out+channel_out


class UpSample(nn.Module):
    """
    上采样模块
    """
    def __init__(self,in_channels, out_channels):
        super().__init__()
        # 转置卷积层 用于上采样
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_layer = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 计算上采样后的特征图 x1 和跳跃连接的特征图 x2 在高度和宽度上的尺寸差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x2.size()[3]

        # 特征图对齐
        x1 = F.pad(x1,[diffX//2,diffX-diffX//2,  # 计算宽度方向左右两侧的填充量
                       diffY//2,diffY-diffY//2])  # 计算高度方向上下两侧的填充量

        # 特征图拼接
        x = torch.cat([x2,x1], dim=1)
        return self.conv_layer(x)


class FinalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 下采样模块
        # 通道数增加 空间维度减少
        self.down1 = DownSample(3,64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        # 注意力模块
        self.attention = AttentionModule(512)

        # 多尺度特征提取与融合模块
        self.fusion = MultiScaleFusion(512)

        # 上采样模块
        self.up1 = UpSample(512,256)
        self.up2 = UpSample(256, 128)
        self.up3 = UpSample(128,64)
        self.up4 = UpSample(64, 32)

        # 输出层
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # 下采样

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # 注意力模块
        x_attention = self.attention(x4)
        # 多尺度特征提取与融合
        x_fused = self.fusion(x_attention)

        # 上采样
        x_up1 = self.up1(x_fused,x3)
        x_up2 = self.up2(x_up1,x2)
        x_up3 = self.up3(x_up2,x1)
        x_up4 = self.up4(x_up3,x)

        # 输出
        output = self.final_conv(x_up4)
        return output


# 测试网络
if __name__ == "__main__":
    model = FinalNetwork()
    input_tensor = torch.randn(1, 3, 512, 512)
    output = model(input_tensor)
    print("Output shape:", output.shape)





































