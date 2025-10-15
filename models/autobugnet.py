
import torch, torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.conv(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, seg_ch=6):
        super().__init__()
        self.d1 = ConvBlock(in_ch, 32)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = ConvBlock(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = ConvBlock(64, 128)
        self.p3 = nn.MaxPool2d(2)
        self.b  = ConvBlock(128, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c3 = ConvBlock(256, 128)
        self.c2 = ConvBlock(128, 64)
        self.c1 = ConvBlock(64, 32)
        self.seg = nn.Conv2d(32, seg_ch, 1)
        # global pooling for multi-label classification
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, seg_ch)
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        b  = self.b(self.p3(d3))
        up3 = self.u3(b)
        c3 = self.c3(torch.cat([up3, d3], dim=1))
        up2 = self.u2(c3)
        c2 = self.c2(up2)
        up1 = self.u1(c2)
        c1 = self.c1(up1)
        seg = self.seg(c1)
        cls = self.cls_head(b)
        return seg, cls
