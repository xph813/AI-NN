import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from task2_operator.functions import get_func_set

# 配置 GPU（兼容 3090，直接拉满算力）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

# 基准配置
BASE_NET_CONFIG = {"hidden_layers": 3, "hidden_units": 50}
BASE_TRAIN_CONFIG = {"epochs": 1000, "display_every": 100, "lr": 1e-3}
GEOM_RANGE = (-1, 1)

# 公共工具：创建目录 + 绘图（英文标题，兼容报告）
def create_result_dir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

def plot_loss_curve(train_losses, test_losses, exp_name, task_dir):
    save_dir = create_result_dir(f"results/{task_dir}/{exp_name}")
    epochs = np.arange(1, len(train_losses)+1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title(f"Loss Convergence - {exp_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_prediction_vs_true(x, y_true, y_pred, exp_name, task_dir, y_label="Function Value"):
    save_dir = create_result_dir(f"results/{task_dir}/{exp_name}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="True Value", linewidth=2)
    plt.plot(x, y_pred, label="Predicted Value", linewidth=2, linestyle="--")
    plt.xlabel("Input")
    plt.ylabel(y_label)
    plt.title(f"Prediction vs True - {exp_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/prediction.png", dpi=300, bbox_inches="tight")
    plt.close()

# 构建 FNN 网络（支持不同激活函数，适配实验需求）
class FNN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_layers=3, hidden_units=50, activation="tanh"):
        super(FNN, self).__init__()
        layers = []
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(self._get_activation(activation))
        # 隐藏层
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(self._get_activation(activation))
        # 输出层
        layers.append(nn.Linear(hidden_units, output_dim))
        
        self.model = nn.Sequential(*layers).to(device)
    
    def _get_activation(self, act_name):
        if act_name == "tanh":
            return nn.Tanh()
        elif act_name == "relu":
            return nn.ReLU()
        elif act_name == "swish":
            return nn.SiLU()
        elif act_name == "gelu":
            return nn.GELU()
        elif act_name == "sigmoid":
            return nn.Sigmoid()
        else:
            return nn.Tanh()
    
    def forward(self, x):
        return self.model(x)

# 数据生成工具（适配不同采样数量）
def generate_data(func, num_train, num_test=50):
    x_train = np.random.uniform(GEOM_RANGE[0], GEOM_RANGE[1], (num_train, 1)).astype(np.float32)
    x_test = np.random.uniform(GEOM_RANGE[0], GEOM_RANGE[1], (num_test, 1)).astype(np.float32)
    y_train = func(x_train).astype(np.float32)
    y_test = func(x_test).astype(np.float32)
    
    # 转换为 PyTorch 张量并移到 GPU
    x_train_tensor = torch.tensor(x_train).to(device)
    y_train_tensor = torch.tensor(y_train).to(device)
    x_test_tensor = torch.tensor(x_test).to(device)
    y_test_tensor = torch.tensor(y_test).to(device)
    
    return (x_train_tensor, y_train_tensor), (x_test_tensor, y_test_tensor)

# 训练函数（记录损失，支持可视化）
def train_model(model, train_data, test_data, epochs, display_every, lr):
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(x_train)
        train_loss = criterion(y_pred_train, y_train)
        train_loss.backward()
        optimizer.step()
        
        # 测试步骤
        model.eval()
        with torch.no_grad():
            y_pred_test = model(x_test)
            test_loss = criterion(y_pred_test, y_test)
        
        # 记录损失
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        # 打印进度
        if epoch % display_every == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}")
    
    return train_losses, test_losses, model

# 实验1：5组激活函数对比
def run_activation_experiment():
    activations = ["tanh", "relu", "swish", "gelu", "sigmoid"]
    func, _ = get_func_set(1)
    num_train = 100
    num_test = 50
    
    for act in activations:
        exp_name = f"Antiderivative_Activation_{act}"
        # 生成数据
        train_data, test_data = generate_data(func, num_train, num_test)
        # 构建模型
        model = FNN(
            hidden_layers=BASE_NET_CONFIG["hidden_layers"],
            hidden_units=BASE_NET_CONFIG["hidden_units"],
            activation=act
        )
        # 训练模型
        train_losses, test_losses, trained_model = train_model(
            model, train_data, test_data,
            epochs=BASE_TRAIN_CONFIG["epochs"],
            display_every=BASE_TRAIN_CONFIG["display_every"],
            lr=BASE_TRAIN_CONFIG["lr"]
        )
        # 可视化
        plot_loss_curve(train_losses, test_losses, exp_name, "task2")
        # 生成预测数据
        x = np.linspace(GEOM_RANGE[0], GEOM_RANGE[1], 1000)[:, None].astype(np.float32)
        x_tensor = torch.tensor(x).to(device)
        trained_model.eval()
        with torch.no_grad():
            y_pred = trained_model(x_tensor).cpu().numpy()
        y_true = func(x)
        plot_prediction_vs_true(x, y_true, y_pred, exp_name, "task2")

# 实验2：5组采样数量对比
def run_sampling_experiment():
    sampling_nums = [30, 50, 100, 200, 500]
    func, _ = get_func_set(1)
    act = "tanh"
    
    for num in sampling_nums:
        exp_name = f"Antiderivative_Sampling_{num}"
        # 生成数据（按当前采样数量）
        train_data, test_data = generate_data(func, num, 50)
        # 构建模型
        model = FNN(
            hidden_layers=BASE_NET_CONFIG["hidden_layers"],
            hidden_units=BASE_NET_CONFIG["hidden_units"],
            activation=act
        )
        # 训练模型
        train_losses, test_losses, trained_model = train_model(
            model, train_data, test_data,
            epochs=BASE_TRAIN_CONFIG["epochs"],
            display_every=BASE_TRAIN_CONFIG["display_every"],
            lr=BASE_TRAIN_CONFIG["lr"]
        )
        # 可视化
        plot_loss_curve(train_losses, test_losses, exp_name, "task2")
        # 生成预测数据
        x = np.linspace(GEOM_RANGE[0], GEOM_RANGE[1], 1000)[:, None].astype(np.float32)
        x_tensor = torch.tensor(x).to(device)
        trained_model.eval()
        with torch.no_grad():
            y_pred = trained_model(x_tensor).cpu().numpy()
        y_true = func(x)
        plot_prediction_vs_true(x, y_true, y_pred, exp_name, "task2")

# 实验3：5组函数复杂度对比
def run_complexity_experiment():
    func_ids = [1, 2, 3, 4, 5]
    act = "tanh"
    num_train = 100
    
    for func_id in func_ids:
        exp_name = f"Antiderivative_Complexity_{func_id}"
        func, _ = get_func_set(func_id)
        # 生成数据
        train_data, test_data = generate_data(func, num_train, 50)
        # 构建模型
        model = FNN(
            hidden_layers=BASE_NET_CONFIG["hidden_layers"],
            hidden_units=BASE_NET_CONFIG["hidden_units"],
            activation=act
        )
        # 训练模型
        train_losses, test_losses, trained_model = train_model(
            model, train_data, test_data,
            epochs=BASE_TRAIN_CONFIG["epochs"],
            display_every=BASE_TRAIN_CONFIG["display_every"],
            lr=BASE_TRAIN_CONFIG["lr"]
        )
        # 可视化
        plot_loss_curve(train_losses, test_losses, exp_name, "task2")
        # 生成预测数据
        x = np.linspace(GEOM_RANGE[0], GEOM_RANGE[1], 1000)[:, None].astype(np.float32)
        x_tensor = torch.tensor(x).to(device)
        trained_model.eval()
        with torch.no_grad():
            y_pred = trained_model(x_tensor).cpu().numpy()
        y_true = func(x)
        plot_prediction_vs_true(x, y_true, y_pred, exp_name, "task2")