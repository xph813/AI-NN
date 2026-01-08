
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

set_random_seed(seed=5)

def sin_2pi_on_grid(x):
    """Computes y = sin(2pi*x)"""
    y = np.sin(2 * np.pi * x)
    return y

num_points = 100
x_original = np.linspace(0, 1, num_points)

# 仅给x加噪声
noise_intensity_x = 0.3  # x的噪声强度
x_noise = np.random.normal(0, noise_intensity_x, len(x_original))  # x的噪声数组
x_noisy = x_original + x_noise  # 加噪后的x（可能乱序）
# 可选：限制x_noisy在[0,1]范围内（避免x超出原定义域）
x_noisy = np.clip(x_noisy, 0, 1)

# 基于加噪后的x计算干净的y（无任何y噪声，仅反映x噪声的影响）
y_clean = sin_2pi_on_grid(x_noisy)

class MLP(nn.Module):
    def __init__(self, hidden_units = 32):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1, hidden_units)
        self.hidden2 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(hidden_units, 1)
    
    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

# 训练数据：加噪的x（乱序） + 干净的y（保持一一映射）
x_tensor = torch.tensor(x_noisy, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y_clean, dtype=torch.float32).view(-1, 1)

# 测试数据：生成新的独立x噪声，y仍为干净值
set_random_seed(seed=1)  # 测试集独立种子
x_noise_test = np.random.normal(0, noise_intensity_x, len(x_original))
x_noisy_test = x_original + x_noise_test
x_noisy_test = np.clip(x_noisy_test, 0, 1)  # 限制范围
y_clean_test = sin_2pi_on_grid(x_noisy_test)  # 测试y仍为干净值
x_test_tensor = torch.tensor(x_noisy_test, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_clean_test, dtype=torch.float32).view(-1, 1)

model = MLP(hidden_units = 32)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

epoch_list = [0]  # 初始epoch
for i in range(10):
    epoch_list.append((i+1) * 100)  # [0,100,200,...,1000]
num_epochs = 1000

check_name = f'./checkpoint_x{str(noise_intensity_x)}/'
os.makedirs(check_name, exist_ok=True)

loss_history = []  # training loss
loss_test = []     # testing loss

for epoch in range(num_epochs):
    if epoch == 0:
        model_file_path = os.path.join(check_name, f'model_{epoch}.pth')
        torch.save(model, model_file_path)
    
    # 训练阶段（x保持原始乱序，保证映射正确）
    model.train()
    optimizer.zero_grad()
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)  # 标签是干净的y
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    # 保存指定epoch的模型
    if (epoch+1) in epoch_list:
        torch.save(model, os.path.join(check_name, f'model_{epoch+1}.pth'))
    
    model.eval()
    with torch.no_grad():
        outputs_test = model(x_test_tensor)  # 测试x是新的加噪x
        loss_t = criterion(outputs_test, y_test_tensor)  # 测试标签是干净y
        loss_test.append(loss_t.item())

np.save(os.path.join(check_name, f'loss_history_{noise_intensity_x}.npy'), loss_history)
np.save(os.path.join(check_name, f'loss_test_{noise_intensity_x}.npy'), loss_test)
np.save(os.path.join(check_name, f'epoch_list.npy'), epoch_list)

plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss', color='#ff6b6b')
plt.plot(loss_test, label='Testing Loss', color='#4ecdc4')
plt.title(f'Loss Curve (Only X Noise, Intensity {noise_intensity_x})')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.ylim(0, 0.7)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(check_name, 'train_test_loss.png'), dpi=300, bbox_inches='tight')
plt.show()

model.eval()
with torch.no_grad():
    predicted_train = model(x_tensor).numpy()
    predicted_test = model(x_test_tensor).numpy()

# ------------ 核心：排序x，同步重排y和预测值（消除折返） ------------
# 训练集排序
idx_train_sorted = np.argsort(x_noisy)  # 获取x_noisy的排序索引
x_train_sorted = x_noisy[idx_train_sorted]
y_train_sorted = y_clean[idx_train_sorted]
pred_train_sorted = predicted_train[idx_train_sorted]

# 测试集排序
idx_test_sorted = np.argsort(x_noisy_test)
x_test_sorted = x_noisy_test[idx_test_sorted]
y_test_sorted = y_clean_test[idx_test_sorted]
pred_test_sorted = predicted_test[idx_test_sorted]

# ===================== 绘制最终预测结果（无折返） =====================
plt.figure(figsize=(10, 5))
# 原始干净x的正弦曲线（参考基准）
plt.plot(x_original, sin_2pi_on_grid(x_original), 
         label='True: sin(2πx)', color='black', linestyle='-', alpha=0.6, linewidth=2)
# 排序后的训练数据点（加噪x + 干净y）
plt.scatter(x_train_sorted, y_train_sorted, 
            label='Train Data (sorted noisy x)', color='green', s=12, alpha=0.8)
# 排序后的测试数据点（新的加噪x + 干净y）
plt.scatter(x_test_sorted, y_test_sorted, 
            label='Test Data (sorted noisy x)', color='orange', s=12, alpha=0.8)
# 排序后的训练集预测曲线（无折返）
plt.plot(x_train_sorted, pred_train_sorted, 
         label='Pred on Train Data (1000 Epochs)', linestyle='--', color="#fc027b", linewidth=1.5)
# 排序后的测试集预测曲线（无折返）
plt.plot(x_test_sorted, pred_test_sorted, 
         label='Pred on Test Data (1000 Epochs)', linestyle='--', color="#00a8ff", linewidth=1.5)
plt.title('True vs Predicted (Sorted X)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(check_name, 'final_prediction_sorted.png'), dpi=300, bbox_inches='tight')
plt.show()

# ===================== 辅助函数：加载模型并预测（排序版） =====================
def predict_with_model(model_path, x_input_tensor, sort_idx):
    """加载模型并预测，返回排序后的预测值"""
    model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        pred = model(x_input_tensor).numpy()
    # 按排序索引重排预测值
    pred_sorted = pred[sort_idx]
    return pred_sorted

# 加载不同epoch的模型，预测并排序
pred_train_epochs = []
for epoch in epoch_list:
    model_path = os.path.join(check_name, f'model_{epoch}.pth')
    pred_sorted = predict_with_model(model_path, x_tensor, idx_train_sorted)
    pred_train_epochs.append(pred_sorted)

# 连续渐变色系（匹配分析代码）
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
# 原始干净x的正弦曲线（参考）
plt.plot(x_original, sin_2pi_on_grid(x_original), 
         label='True: sin(2πx) (clean x)', color='black', linewidth=2, alpha=0.6)
# 排序后的训练数据点
plt.scatter(x_train_sorted, y_train_sorted, 
            label='Train Data (sorted noisy x)', color='green', s=8, alpha=0.6)
# 不同epoch的预测曲线（排序后无折返）
for i, epoch in enumerate(epoch_list):
    plt.plot(x_train_sorted, pred_train_epochs[i], 
             label=f'Pred: {epoch} Epochs', color=colors[i], alpha=0.7, linewidth=1.2)
plt.title(f'Prediction Evolution (Sorted X, X Noise {noise_intensity_x})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(check_name, 'prediction_evolution_sorted.png'), dpi=300, bbox_inches='tight')
plt.show()

# ===================== 额外验证：输出x排序前后的对比 =====================
print("=== X噪声验证 ===")
print(f"原始x是否严格递增：{np.all(np.diff(x_original) > 0)}")
print(f"加噪x是否严格递增：{np.all(np.diff(x_noisy) > 0)}")
print(f"排序后x是否严格递增：{np.all(np.diff(x_train_sorted) > 0)}")
print(f"x噪声强度：{noise_intensity_x}，x_noisy范围：[{x_noisy.min():.4f}, {x_noisy.max():.4f}]")