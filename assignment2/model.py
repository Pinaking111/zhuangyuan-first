
import torch
import torch.nn as nn

class TinyCifarCNN(nn.Module):
    """
    Input:  RGB 64x64
    Conv(3->16, k3,s1,p1) -> ReLU -> MaxPool(2,2)
    Conv(16->32, k3,s1,p1) -> ReLU -> MaxPool(2,2)
    Flatten -> FC(100) -> ReLU -> FC(10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 64x64 -> 64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 64x64 -> 32x32

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 32x32 -> 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)                  # 32x32 -> 16x16
        )
        # 32 channels @ 16x16 -> 32*16*16 = 8192
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
