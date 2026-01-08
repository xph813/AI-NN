
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os

def set_random_seed(seed=42):
    """固定所有随机数种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

noise_intensity_x = 0.1  # x的噪声强度
noise_intensity_y = 0.4   # y的噪声强度（独立于x噪声）
# 目录路径（包含x、y噪声强度，避免文件覆盖）
check_name = f'./checkpoint_x{noise_intensity_x}_y{noise_intensity_y}/'
os.makedirs(check_name, exist_ok=True)

# 调用种子设置函数（全局基础种子）
set_random_seed(seed=5)

# ===================== 数据生成（x、y均加独立噪声） =====================
def sin_2pi_on_grid(x):
    """Computes y = sin(2pi*x)"""
    return np.sin(2 * np.pi * x)

# 1. 生成原始干净数据
num_points = 100
x_original = np.linspace(0, 1, num_points)
y_clean = sin_2pi_on_grid(x_original)  # y基于干净x计算，避免耦合

# 2. 给x加独立噪声（固定种子）
set_random_seed(seed=5)  # x噪声专属种子
x_noise = np.random.normal(0, noise_intensity_x, len(x_original))
x_noisy = x_original + x_noise
x_noisy = np.clip(x_noisy, 0, 1)  # 限制x在[0,1]定义域内

# 3. 给y加独立噪声（固定新种子，与x噪声独立）
set_random_seed(seed=6)  # y噪声专属种子（≠x噪声种子）
y_noise = np.random.normal(0, noise_intensity_y, len(y_clean))
y_noisy = y_clean + y_noise  # y噪声独立于x噪声

# 4. 测试集：生成新的独立x/y噪声（泛化性测试）
set_random_seed(seed=1)  # 测试集专属种子
x_noise_test = np.random.normal(0, noise_intensity_x, len(x_original))
x_noisy_test = x_original + x_noise_test
x_noisy_test = np.clip(x_noisy_test, 0, 1)

y_noise_test = np.random.normal(0, noise_intensity_y, len(y_clean))
y_noisy_test = y_clean + y_noise_test  # 测试集y噪声也独立

# ===================== 定义MLP模型（与单噪声场景一致） =====================
class MLP(nn.Module):
    def __init__(self, hidden_units=32):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1, hidden_units)
        self.hidden2 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(hidden_units, 1)
    
    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

# 训练数据：加噪x + 加噪y
x_tensor = torch.tensor(x_noisy, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y_noisy, dtype=torch.float32).view(-1, 1)

# 测试数据：新的加噪x + 新的加噪y
x_test_tensor = torch.tensor(x_noisy_test, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_noisy_test, dtype=torch.float32).view(-1, 1)

model = MLP(hidden_units=32)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

epoch_list = [0]  # 初始epoch
for i in range(10):
    epoch_list.append((i+1) * 100)  # [0,100,200,...,1000]
num_epochs = 1000

loss_history = []  # training loss
loss_test = []     # testing loss

for epoch in range(num_epochs):
    if epoch == 0:
        torch.save(model, os.path.join(check_name, f'model_{epoch}.pth'))
    
    model.train()
    optimizer.zero_grad()
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    if (epoch+1) in epoch_list:
        torch.save(model, os.path.join(check_name, f'model_{epoch+1}.pth'))
    
    model.eval()
    with torch.no_grad():
        outputs_test = model(x_test_tensor)
        loss_t = criterion(outputs_test, y_test_tensor)
        loss_test.append(loss_t.item())

loss_history_filename = f'loss_history_x{noise_intensity_x}_y{noise_intensity_y}.npy'
loss_test_filename = f'loss_test_x{noise_intensity_x}_y{noise_intensity_y}.npy'
np.save(os.path.join(check_name, loss_history_filename), loss_history)
np.save(os.path.join(check_name, loss_test_filename), loss_test)
np.save(os.path.join(check_name, 'epoch_list.npy'), epoch_list)

plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss', color='#ff6b6b')
plt.plot(loss_test, label='Testing Loss', color='#4ecdc4')
plt.title(f'Loss Curve (X Noise={noise_intensity_x}, Y Noise={noise_intensity_y})')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.ylim(0, 1.5)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(check_name, 'train_test_loss.png'), dpi=300, bbox_inches='tight')
plt.show()

model.eval()
with torch.no_grad():
    predicted_train = model(x_tensor).numpy()
    predicted_test = model(x_test_tensor).numpy()

# 训练集排序（x从小到大，同步重排y和预测值）
idx_train_sorted = np.argsort(x_noisy)
x_train_sorted = x_noisy[idx_train_sorted]
y_train_sorted = y_noisy[idx_train_sorted]
pred_train_sorted = predicted_train[idx_train_sorted]

# 测试集排序
idx_test_sorted = np.argsort(x_noisy_test)
x_test_sorted = x_noisy_test[idx_test_sorted]
y_test_sorted = y_noisy_test[idx_test_sorted]
pred_test_sorted = predicted_test[idx_test_sorted]

# ===================== 绘制最终预测结果 =====================
plt.figure(figsize=(10, 5))
# 原始干净曲线（参考基准）
plt.plot(x_original, y_clean, label='True: sin(2πx) (clean x+y)', color='black', alpha=0.6, linewidth=2)
# 训练数据点（x_noisy + y_noisy）
plt.scatter(x_train_sorted, y_train_sorted, label='Train Data (noisy x+y)', color='green', s=12, alpha=0.8)
# 测试数据点（new noisy x+y）
plt.scatter(x_test_sorted, y_test_sorted, label='Test Data (new noisy x+y)', color='orange', s=12, alpha=0.8)
# 训练集预测曲线（无折返）
plt.plot(x_train_sorted, pred_train_sorted, label='Pred on Train Data', linestyle='--', color="#fc027b")
# 测试集预测曲线（无折返）
plt.plot(x_test_sorted, pred_test_sorted, label='Pred on Test Data', linestyle='--', color="#00a8ff")
plt.title(f'True vs Predicted (X Noise={noise_intensity_x}, Y Noise={noise_intensity_y})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(check_name, 'final_prediction_sorted.png'), dpi=300, bbox_inches='tight')
plt.show()

def predict_with_model(model_path, x_input_tensor, sort_idx):
    """加载模型并返回排序后的预测值"""
    model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        pred = model(x_input_tensor).numpy()
    return pred[sort_idx]

pred_train_epochs = []
for epoch in epoch_list:
    model_path = os.path.join(check_name, f'model_{epoch}.pth')
    pred_sorted = predict_with_model(model_path, x_tensor, idx_train_sorted)
    pred_train_epochs.append(pred_sorted)

color_list = [
    "#00d8b6", "#00c4e0", "#00a8ff", "#4080ff", "#7060ff",
    "#9040ff", "#b020ff", "#e040d0", "#ff5080", "#ff8040", "#ffb020"
]
color_map = {epoch: color_list[i] for i, epoch in enumerate(epoch_list)}

plt.figure(figsize=(12, 6))
# 干净参考曲线
plt.plot(x_original, y_clean, label='True: sin(2πx) (clean x+y)', color='black', linewidth=2, alpha=0.6)
# 训练数据点
plt.scatter(x_train_sorted, y_train_sorted, label='Train Data (noisy x+y)', color='green', s=8, alpha=0.6)
# 不同epoch预测曲线
for i, epoch in enumerate(epoch_list):
    plt.plot(x_train_sorted, pred_train_epochs[i], label=f'Pred: {epoch} Epochs', color=color_map[epoch], alpha=0.7)
plt.title(f'Prediction Evolution (X Noise={noise_intensity_x}, Y Noise={noise_intensity_y})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(check_name, 'prediction_evolution_sorted.png'), dpi=300, bbox_inches='tight')
plt.show()

# ===================== 验证信息输出 =====================
print("=== 双噪声训练完成 ===")
print(f"x噪声强度: {noise_intensity_x}, y噪声强度: {noise_intensity_y}")
print(f"模型文件保存至: {check_name}")
print(f"训练集x是否严格递增（排序后）: {np.all(np.diff(x_train_sorted) > 0)}")