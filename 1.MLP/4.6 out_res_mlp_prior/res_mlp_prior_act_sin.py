import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os

# -------------------------- 1. 固定随机种子 --------------------------
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_random_seed(seed=5)

# -------------------------- 2. 自定义周期激活函数（核心修改） --------------------------
class SinActivation(nn.Module):
    """自定义正弦激活函数（周期激活），替代非周期的Tanh"""
    def forward(self, x):
        return torch.sin(x)  # 利用sin的天然周期性，让网络层输出具备周期属性

# -------------------------- 3. 生成sin周期数据 --------------------------
def sin_2pi_on_grid(x, period):
    """Computes y = sin(2pi*x*period) （修正原公式：原代码period*2πx等价于2πx*period，更符合周期定义）"""
    y = np.sin(2 * np.pi * period * x)
    return y

period = 4  # 目标周期：sin(8πx)，周期为1/4
num_points = 100
x = np.linspace(0, 1, num_points)
y = sin_2pi_on_grid(x, period)

# 加噪声
noise_intensity = 0.3
noise = np.random.normal(0, noise_intensity, len(y))
y_noise = y + noise

# -------------------------- 4. 带周期激活的残差网络（核心修改） --------------------------
class ResBlock(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        layers = []
        for _ in range(4):
            layers.append(nn.Linear(hidden_units, hidden_units))
            # 核心修改：替换Tanh为自定义Sin激活
            layers.append(SinActivation())  
        self.net = nn.Sequential(*layers)
        self.shortcut = nn.Identity()
    
    def forward(self, x):
        return self.net(x) + self.shortcut(x)

class ResMlp(nn.Module):
    def __init__(self, blocks, hidden_units=32):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, hidden_units))
        # 输入层也替换为Sin激活（保持周期一致性）
        layers.append(SinActivation())  

        for _ in range(blocks):
            layers.append(ResBlock(hidden_units))
        
        layers.append(nn.Linear(hidden_units, 1))
        # 输出层无激活（回归任务）
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# -------------------------- 5. 数据预处理 --------------------------
USE_NOISE = True
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
if USE_NOISE:
    y_tensor = torch.tensor(y_noise, dtype=torch.float32).view(-1, 1)
else:
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 初始化模型
blocks = 200
layers = 4 * blocks + 2
model = ResMlp(blocks, hidden_units=32)

# 优化器（微调学习率：Sin激活梯度更稳定，稍调高学习率）
criterion = nn.MSELoss()
lr = 5e-4  # 原1e-4 → 5e-4
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)  # 加权重衰减抑制过拟合

# 生成测试噪声数据
set_random_seed(seed=1)
noise_test = np.random.normal(0, noise_intensity, len(y))
y_noise_test = y + noise_test
y_noise_test_tensor = torch.tensor(y_noise_test, dtype=torch.float32).view(-1, 1)

# -------------------------- 6. 模型训练 --------------------------
epoch_list = [0]
for i in range(10):
    epoch_list.append((i+1) * 100)
num_epochs = 1000

# 保存路径
check_name = f'./checkpoint_{layers}_layer_{period}_period_sin_activation/'
os.makedirs(check_name, exist_ok=True)

# 记录损失
loss_history = []  # 训练损失
loss_test = []     # 测试损失

for epoch in range(num_epochs):
    # 保存初始模型
    if epoch == 0:
        model_file_path = os.path.join(check_name, f'model_{epoch}.pth')
        torch.save(model, model_file_path)
    
    # 训练阶段
    model.train()
    optimizer.zero_grad()
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    # 保存指定epoch的模型
    if (epoch+1) in epoch_list:
        torch.save(model, os.path.join(check_name, f'model_{epoch+1}.pth'))
    
    # 测试阶段（无梯度）
    model.eval()
    with torch.no_grad():
        outputs_test = model(x_tensor)
        loss_t = criterion(outputs_test, y_noise_test_tensor)
        loss_test.append(loss_t.item())

# 保存损失数据
np.save(os.path.join(check_name, f'loss_history_{noise_intensity}.npy'), loss_history)
np.save(os.path.join(check_name, f'loss_test_{noise_intensity}.npy'), loss_test)
np.save(os.path.join(check_name, f'epoch_list.npy'), epoch_list)

# -------------------------- 7. 绘制损失曲线 --------------------------
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss', color='#ff6b6b')
plt.plot(loss_test, label='Testing Loss', color='#4ecdc4')
plt.title(f'Loss Curve (Sin Activation, num_points: {num_points})')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.ylim(0, 0.7)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(check_name, 'train_test_loss.png'), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 8. 绘制最终拟合结果 --------------------------
model.eval()
with torch.no_grad():
    predicted = model(x_tensor).numpy()

plt.figure(figsize=(10, 5))
plt.plot(x, y, label=f'y = sin(2π*{period}x)', color='black')
plt.plot(x, predicted, label='Pred Final', linestyle='--', color="#fc027b")
plt.plot(x, y_noise, label='y+noise (train)', color='green', alpha=0.5)
plt.plot(x, y_noise_test, label='y+noise (test)', color='orange', alpha=0.5)
plt.title('True vs Predicted (Final Model, Sin Activation)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# -------------------------- 9. 绘制不同epoch的预测演化 --------------------------
def diff_model(model):
    model.eval()
    with torch.no_grad():
        predicted = model(x_tensor).numpy()
    return predicted

pred = []
for i in epoch_list:
    model_new = torch.load(os.path.join(check_name, f'model_{i}.pth'), weights_only=False)
    pred.append(diff_model(model_new))

# 颜色映射
color_map = {
    epoch_list[0]: "#00d8b6",
    epoch_list[1]: "#00c4e0",
    epoch_list[2]: "#00a8ff",
    epoch_list[3]: "#4080ff",
    epoch_list[4]: "#7060ff",
    epoch_list[5]: "#9040ff",
    epoch_list[6]: "#b020ff",
    epoch_list[7]: "#e040d0",
    epoch_list[8]: "#ff5080",
    epoch_list[9]: "#ff8040",
    epoch_list[10]: "#ffb020"
}
colors = [color_map[e] for e in epoch_list]

plt.figure(figsize=(12, 6))
plt.plot(x, y, label='True: sin(8πx)', color='black', linewidth=2)
plt.plot(x, y_noise, label='Train Data: y+noise', color='green', alpha=0.3)
for i in range(len(epoch_list)):
    plt.plot(x, pred[i], label=f'Pred: {epoch_list[i]} Epochs', color=colors[i], alpha=0.7)
plt.title(f'Prediction Evolution (Sin Activation, Noise {noise_intensity})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(check_name, 'prediction_evolution.png'), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 10. 外推测试（核心验证：周期外推） --------------------------
num_points_ext = 300
x_ext = np.linspace(-1, 2, num_points_ext)  # 外推区间：-1到2（覆盖3个周期）
y_ext = sin_2pi_on_grid(x_ext, period)

# 外推测试加噪声
noise_intensity_ext = 0.4
noise_test_ext = np.random.normal(0, noise_intensity_ext, len(y_ext))
y_noise_test_ext = y_ext + noise_test_ext

# 转换为tensor
x_ext_tensor = torch.tensor(x_ext, dtype=torch.float32).view(-1, 1)

# 外推预测
model.eval()
with torch.no_grad():
    predicted_ext = model(x_ext_tensor).numpy()

# 绘制外推结果
plt.figure(figsize=(10, 5))
plt.plot(x_ext, y_ext, label=f'y = sin(2π*{period}x)', color='black', linewidth=2)
plt.plot(x_ext, predicted_ext, label='Pred Final (1000 Epochs, Sin Activation)', linestyle='--', color="#fc027b")
plt.plot(x_ext, y_noise_test_ext, label='y+noise (test)', color='orange', alpha=0.5)
# 标记训练区间
plt.axvspan(0, 1, color='gray', alpha=0.1, label='Training Interval [0,1]')
plt.title(f'Extrapolation Test (Sin Activation, num_points={num_points_ext})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(check_name, 'prediction[-1, 2]_sin_activation.png'), dpi=300, bbox_inches='tight')
plt.show()