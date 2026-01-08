# 训练完后运行该可视化（这个是AI写的，感觉不是很好，建议改）
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from data_utils_model import STNNet, DEVICE, get_test_dataloader

# ===================== 可视化工具函数 =====================
# 修正：添加model和test_loader两个形参
def visualize_stn(model, test_loader):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(DEVICE)

        # ========== 新增：给图像加轻微扭曲（让STN变换可见） ==========
        import random
        from torchvision.transforms import functional as TF
        distorted_data = []
        for img in data:
            # 随机小角度旋转（-10~10度）+ 轻微平移
            angle = random.randint(-60, 60)
            dx, dy = random.randint(-2, 2), random.randint(-2, 2)
            img_dist = TF.affine(img, angle=angle, translate=(dx/28, dy/28), scale=1, shear=0)
            distorted_data.append(img_dist)
        distorted_data = torch.stack(distorted_data).to(DEVICE)
        # ============================================================

        input_tensor = distorted_data.cpu()  # 改用扭曲后的图像作为输入
        transformed_input_tensor = model.stn(distorted_data).cpu()  # STN矫正扭曲图像

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2, figsize=(12, 6))  # 新增：调整画布大小，避免挤在一起
        axarr[0].imshow(in_grid[:, :, 0], cmap='gray')  # 改动：取单通道+灰度显示
        axarr[0].set_title('Distorted Images')  # 改标题：明确是扭曲后的图像
        axarr[0].axis('off')  # 新增：隐藏坐标轴，更清晰

        axarr[1].imshow(out_grid[:, :, 0], cmap='gray')  # 改动：取单通道+灰度显示
        axarr[1].set_title('STN Corrected Images')  # 改标题：明确是STN矫正后
        axarr[1].axis('off')  # 新增：隐藏坐标轴

        plt.tight_layout()  # 新增：自动调整布局
        plt.show()

# ========== 修复convert_image_np函数（仅改这一处，适配MNIST） ==========
def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    # 改动：用MNIST的归一化参数反归一化
    mean = np.array([0.1307])
    std = np.array([0.3081])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# ===================== 加载模型并可视化 =====================
if __name__ == '__main__':
    # 1. 初始化模型
    model = STNNet().to(DEVICE)
    
    # 2. 加载训练好的权重
    SAVE_PATH = "./stn_mnist_model.pth"
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载模型权重: {SAVE_PATH}")
    
    # 3. 获取测试数据
    test_loader = get_test_dataloader(batch_size=64)
    
    # 4. 执行可视化（调用格式正确）
    visualize_stn(model, test_loader)