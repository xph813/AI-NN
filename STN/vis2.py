# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# 1. 完全复制你的网络依赖（DEVICE、TRANSFORM、get_test_dataloader）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
])

def get_test_dataloader(batch_size=64):
    test_loader = torch.utils.data.DataLoader(datasets.MNIST("./data", train=False, transform=TRANSFORM), batch_size=batch_size, shuffle=False)
    return test_loader

# 2. 完全复制你的STNNet网络（一字不改，确保匹配）
class STNNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1.注意力层（=仿射变换）
        ## 1.1特征提取
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

# 3. 原作者的convert_image_np（完全不变，保持极简风格）
def convert_image_np(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# 4. 原作者的visualize_stn（仅加必要参数，核心逻辑不变）
def visualize_stn(model, test_loader):
    with torch.no_grad():
        data = next(iter(test_loader))[0].to(DEVICE)
        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor))

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')
        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

# 5. 核心执行逻辑（极简，无冗余）
if __name__ == '__main__':
    # 初始化你的STNNet模型
    model = STNNet().to(DEVICE)

    # 加载权重（提取model_state_dict，解决嵌套问题）
    checkpoint = torch.load("./stn_mnist_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 获取你的测试数据加载器
    test_loader = get_test_dataloader(batch_size=64)

    # 执行可视化
    visualize_stn(model, test_loader)

    # 原作者的显示逻辑
    plt.ioff()
    plt.show()