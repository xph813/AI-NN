
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_random_seed(seed=5)

def sin_2pi_on_grid(x):
    """Computes y = sin(2pi*x) on a uniform grid from 0 to 1."""
    y = np.sin(2 * np.pi * x)
    return y

num_points = 50
x = np.linspace(0, 1, num_points)
y = sin_2pi_on_grid(x)

noise_intensity = 0.3
noise = np.random.normal(0, noise_intensity, len(y))
# noise = np.random.uniform(low=-noise_intensity, high=noise_intensity, size=len(y))
y_noise = y + noise

class MLP(nn.Module):

    def __init__(self, num_layers = 8, hidden_units=32, in_dim = 1, out_dim = 1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_units))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_units, out_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# 网络层数
layers = 60

USE_NOISE = True
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
if USE_NOISE:
    y_tensor = torch.tensor(y_noise, dtype=torch.float32).view(-1, 1)
else:
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

model = MLP(num_layers=layers, hidden_units = 32)
criterion = nn.MSELoss()
lr = 1e-4
optimizer = optim.AdamW(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr)

set_random_seed(seed=1)
noise_intensity_test = noise_intensity
# noise_test = np.random.uniform(low=-noise_intensity, high=noise_intensity, size=len(y))
noise_test = np.random.normal(0, noise_intensity, len(y))

y_noise_test = y + noise_test
y_noise_test_tensor = torch.tensor(y_noise_test, dtype=torch.float32).view(-1, 1)  # 预生成tensor提升效率

epoch_list = [0]
for i in range(10):
    epoch_list.append((i+1) * 1000)
num_epochs = 10000

check_name = f'./checkpoint_{layers}_no_res/'
os.makedirs(check_name, exist_ok=True)

loss_history = []  # training loss
loss_test = []     # testing loss

for epoch in range(num_epochs):
    if epoch == 0:
        model_file_path = os.path.join(check_name, f'model_{epoch}.pth')
        torch.save(model, model_file_path)
    
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
        outputs_test = model(x_tensor)
        loss_t = criterion(outputs_test, y_noise_test_tensor)
        loss_test.append(loss_t.item())

np.save(os.path.join(check_name, f'loss_history_{noise_intensity}.npy'), loss_history)
np.save(os.path.join(check_name, f'loss_test_{noise_intensity}.npy'), loss_test)
np.save(os.path.join(check_name, f'epoch_list.npy'), epoch_list)  # 保存epoch列表

plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss', color='#ff6b6b')
plt.plot(loss_test, label='Testing Loss', color='#4ecdc4')
plt.title(f'Loss Curve (num_points: {num_points})')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.ylim(0, 0.7)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(check_name, 'train_test_loss.png'), dpi=300, bbox_inches='tight')
plt.show()

model.eval()
with torch.no_grad():
    predicted = model(x_tensor).numpy()

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='y = sin(2πx)', color='black')
plt.plot(x, predicted, label='Pred Final', linestyle='--', color="#fc027b")
plt.plot(x, y_noise, label='y+noise (train)', color='green', alpha=0.5)
plt.plot(x, y_noise_test, label='y+noise (test)', color='orange', alpha=0.5)
plt.title('True vs Predicted (Final Model)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

def diff_model(model):
    model.eval()
    with torch.no_grad():
        predicted = model(x_tensor).numpy()
    return predicted

pred = []
for i in epoch_list:
    model_new = torch.load(os.path.join(check_name, f'model_{i}.pth'), weights_only=False)
    pred.append(diff_model(model_new))

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
plt.plot(x, y, label='True: sin(2πx)', color='black', linewidth=2)
plt.plot(x, y_noise, label='Train Data: y+noise', color='green', alpha=0.3)
for i in range(len(epoch_list)):
    plt.plot(x, pred[i], label=f'Pred: {epoch_list[i]} Epochs', color=colors[i], alpha=0.7)
plt.title(f'Prediction Evolution (Noise  {noise_intensity})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(check_name, 'prediction_evolution.png'), dpi=300, bbox_inches='tight')
plt.show()