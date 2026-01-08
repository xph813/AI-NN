# 定义网络
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
])

# 可视化用
def get_test_dataloader(batch_size=64):
    test_loader = torch.utils.data.DataLoader(datasets.MNIST("./data", train=False, transform=TRANSFORM), batch_size=batch_size, shuffle=False)
    return test_loader

class STNNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1.注意力层（=仿射变换）
        ## 1.1特征提取
        # 原作者没加padding，Linear输入难算。这个经过两个最大池化，直接28 / 4 = 7
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        
        ## 1.2.回归学习仿射变换的参数theta
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # 2.平常的分类网络
        ## 2.1特征提取
        self.feature = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        
        ## 2.2分类头head
        self.classifier = nn.Sequential(
            nn.Linear(20 * 7 * 7, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10)
        )

    def stn(self, x):
        loc_feat = self.localization(x)
        theta = self.fc_loc(loc_feat.flatten(1))
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x_transformed = F.grid_sample(x, grid)
        return x_transformed

    def forward(self, x):
        x = self.stn(x)
        feat = self.feature(x)
        logits = self.classifier(feat.flatten(1))
        return F.log_softmax(logits, dim=1)