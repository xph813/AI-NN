'''
提取MLP模型权重/偏置分布 + L1/L2范数 + 参数量统计 + 损失可视化（适配x+y双噪声）
核心适配：双噪声参数 + 路径匹配 + 标题标注
'''
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.stats import skew, kurtosis, gaussian_kde
from x_y_noise import noise_intensity_x, noise_intensity_y
# ===================== 配置项（与训练代码严格对齐） =====================
# ！！！需手动设置为训练代码中使用的噪声强度 ！！！
# noise_intensity_x = 0.05  # x的噪声强度
# noise_intensity_y = 0.4   # y的噪声强度
# 路径配置（匹配训练代码的双噪声目录）
check_name = f'./checkpoint_x{noise_intensity_x}_y{noise_intensity_y}/'
pic_save_dir = f'weight_plots_x{noise_intensity_x}_y{noise_intensity_y}'

# 全局绘图配置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# 加载epoch列表
epoch_list = np.load(os.path.join(check_name, 'epoch_list.npy')).tolist()

# ===================== 1. 定义模型结构（与训练时完全一致） =====================
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

# ===================== 2. 核心函数1：提取参数分布统计 =====================
def extract_param_distribution(model_path):
    model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
    model.eval()
    
    param_dict = {'weight': {}, 'bias': {}}
    stats_dict = {'weight': {}, 'bias': {}}
    global_stats = {'weight': {}, 'bias': {}}
    all_params_flatten = np.array([])
    
    for name, param in model.named_parameters():
        param_np = param.detach().cpu().numpy()
        param_flatten = param_np.flatten()
        all_params_flatten = np.concatenate([all_params_flatten, param_flatten])
        
        if 'weight' in name:
            param_type = 'weight'
        elif 'bias' in name:
            param_type = 'bias'
        else:
            continue
        
        param_dict[param_type][name] = param_flatten
        stats_dict[param_type][name] = {
            'mean': np.mean(param_flatten),
            'std': np.std(param_flatten),
            'var': np.var(param_flatten),
            'min': np.min(param_flatten),
            'max': np.max(param_flatten),
            'median': np.median(param_flatten),
            'skewness': skew(param_flatten),
            'kurtosis': kurtosis(param_flatten)
        }
    
    # 全局统计（weight/bias/全参数）
    for param_type in ['weight', 'bias']:
        global_flatten = np.concatenate(list(param_dict[param_type].values()))
        global_stats[param_type] = {
            'mean': np.mean(global_flatten),
            'std': np.std(global_flatten),
            'var': np.var(global_flatten),
            'min': np.min(global_flatten),
            'max': np.max(global_flatten),
            'median': np.median(global_flatten),
            'skewness': skew(global_flatten),
            'kurtosis': kurtosis(global_flatten),
            'flatten': global_flatten
        }
    
    global_stats['all_params'] = {
        'mean': np.mean(all_params_flatten),
        'std': np.std(all_params_flatten),
        'var': np.var(all_params_flatten),
        'min': np.min(all_params_flatten),
        'max': np.max(all_params_flatten),
        'median': np.median(all_params_flatten),
        'skewness': skew(all_params_flatten),
        'kurtosis': kurtosis(all_params_flatten),
        'flatten': all_params_flatten
    }
    
    return param_dict, stats_dict, global_stats, model

# ===================== 3. 核心函数2：计算范数 =====================
def calculate_norm(param_dict, all_params_flatten):
    l1_norm_dict = {'weight': {}, 'bias': {}}
    l2_norm_dict = {'weight': {}, 'bias': {}}
    
    # weight/bias范数
    for param_type in ['weight', 'bias']:
        total_l1 = 0.0
        total_l2_sq = 0.0
        total_count = 0
        layer_l1 = {}
        layer_l2 = {}
        
        for layer_name, param_flatten in param_dict[param_type].items():
            l1 = np.sum(np.abs(param_flatten))
            l2 = np.sqrt(np.sum(np.square(param_flatten)))
            layer_l1[layer_name] = l1
            layer_l2[layer_name] = l2
            
            total_l1 += l1
            total_l2_sq += np.sum(np.square(param_flatten))
            total_count += len(param_flatten)
        
        total_l2 = np.sqrt(total_l2_sq)
        avg_l1 = total_l1 / total_count if total_count > 0 else 0
        avg_l2 = total_l2 / np.sqrt(total_count) if total_count > 0 else 0
        
        l1_norm_dict[param_type] = layer_l1
        l1_norm_dict[param_type]['total'] = total_l1
        l1_norm_dict[param_type]['avg'] = avg_l1
        
        l2_norm_dict[param_type] = layer_l2
        l2_norm_dict[param_type]['total'] = total_l2
        l2_norm_dict[param_type]['avg'] = avg_l2
    
    # 全参数范数
    all_norm_dict = {
        'l1': np.sum(np.abs(all_params_flatten)),
        'l2': np.sqrt(np.sum(np.square(all_params_flatten)))
    }
    
    return l1_norm_dict, l2_norm_dict, all_norm_dict

# ===================== 4. 核心函数3：计算参数量 =====================
def calculate_param_count(model):
    param_count_dict = {
        'total_params': 0,
        'trainable_params': 0,
        'layer_params': {},
        'weight_total': 0,
        'bias_total': 0
    }
    
    for name, param in model.named_parameters():
        param_num = param.numel()
        param_count_dict['total_params'] += param_num
        if param.requires_grad:
            param_count_dict['trainable_params'] += param_num
        param_count_dict['layer_params'][name] = param_num
        
        if 'weight' in name:
            param_count_dict['weight_total'] += param_num
        elif 'bias' in name:
            param_count_dict['bias_total'] += param_num
    
    return param_count_dict

# ===================== 5. 辅助打印函数 =====================
def print_l1_norm_results(l1_norm_dict, epoch):
    print(f"\n===== {epoch} Epoch L1范数结果（Weight/Bias分开） =====")
    for param_type in ['weight', 'bias']:
        print(f"\n【{param_type.capitalize()} L1范数】")
        for layer_name, l1 in l1_norm_dict[param_type].items():
            if layer_name not in ['total', 'avg']:
                print(f"  层 {layer_name}: L1 = {l1:.4f}")
        print(f"  全局(total): L1 = {l1_norm_dict[param_type]['total']:.4f}")
        print(f"  平均(avg): L1 = {l1_norm_dict[param_type]['avg']:.6f}")

def print_l2_norm_results(l2_norm_dict, epoch):
    print(f"\n===== {epoch} Epoch L2范数结果（Weight/Bias分开） =====")
    for param_type in ['weight', 'bias']:
        print(f"\n【{param_type.capitalize()} L2范数】")
        for layer_name, l2 in l2_norm_dict[param_type].items():
            if layer_name not in ['total', 'avg']:
                print(f"  层 {layer_name}: L2 = {l2:.4f}")
        print(f"  全局(total): L2 = {l2_norm_dict[param_type]['total']:.4f}")
        print(f"  平均(avg): L2 = {l2_norm_dict[param_type]['avg']:.6f}")

def print_all_params_norm_results(all_norm_dict, epoch):
    print(f"\n===== {epoch} Epoch 全参数范数结果（Weight+Bias合并） =====")
    print(f"  全参数L1范数: {all_norm_dict['l1']:.4f}")
    print(f"  全参数L2范数: {all_norm_dict['l2']:.4f}")

def print_param_count(param_count_dict, epoch):
    print(f"\n===== {epoch} Epoch 模型参数量统计 =====")
    print(f"总参数量: {param_count_dict['total_params']:,}")
    print(f"可训练参数量: {param_count_dict['trainable_params']:,}")
    print(f"Weight总参数: {param_count_dict['weight_total']:,}")
    print(f"Bias总参数: {param_count_dict['bias_total']:,}")
    print("\n【分层参数量】")
    for layer_name, num in param_count_dict['layer_params'].items():
        print(f"  层 {layer_name}: {num:,} 个参数")

# ===================== 6. 可视化函数1：参数分布整合 =====================
def plot_param_distribution_combined(global_stats_dict, save_fig=False):
    color_list = [
        "#00d8b6", "#00c4e0", "#00a8ff", "#4080ff", "#7060ff",
        "#9040ff", "#b020ff", "#e040d0", "#ff5080", "#ff8040", "#ffb020"
    ]
    color_map = {epoch: color_list[i] for i, epoch in enumerate(epoch_list)}
    label_map = {epoch: f'{epoch} Epochs' for epoch in epoch_list}
    
    fig, axes = plt.subplots(3, 2, figsize=(30, 24))
    fig.suptitle(
        f'Parameter Distribution (X Noise={noise_intensity_x}, Y Noise={noise_intensity_y})',
        fontsize=22, y=0.98
    )
    
    param_types = ['weight', 'bias', 'all_params']
    param_titles = ['Weight Parameters', 'Bias Parameters', 'All Parameters (Weight + Bias)']
    
    for row_idx, (param_type, title) in enumerate(zip(param_types, param_titles)):
        ax_hist = axes[row_idx, 0]
        ax_box = axes[row_idx, 1]
        
        # 直方图+KDE
        for epoch in epoch_list:
            param_flatten = global_stats_dict[epoch][param_type]['flatten']
            color = color_map[epoch]
            label = label_map[epoch]
            
            ax_hist.hist(param_flatten, bins='auto', density=True, alpha=0.5, color=color, label=label)
            x_range = np.linspace(min(param_flatten), max(param_flatten), 200)
            kde = gaussian_kde(param_flatten)
            ax_hist.plot(x_range, kde(x_range), color=color, linewidth=2)
        
        ax_hist.set_title(f'{title} - Histogram + KDE', fontsize=18)
        ax_hist.set_xlabel('Parameter Value', fontsize=14)
        ax_hist.set_ylabel('Density', fontsize=14)
        ax_hist.legend(fontsize=12, loc='upper right')
        ax_hist.grid(alpha=0.3)
        
        # 箱线图
        box_data = [global_stats_dict[epoch][param_type]['flatten'] for epoch in epoch_list]
        box_labels = [label_map[epoch] for epoch in epoch_list]
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], color_map.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        ax_box.set_title(f'{title} - Box Plot', fontsize=18)
        ax_box.set_xlabel('Epoch', fontsize=14)
        ax_box.set_ylabel('Parameter Value', fontsize=14)
        ax_box.grid(alpha=0.3)
        ax_box.tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_fig:
        os.makedirs(pic_save_dir, exist_ok=True)
        plt.savefig(f'{pic_save_dir}/param_distribution_combined.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===================== 7. 可视化函数2：范数趋势 =====================
def plot_norm_trend(l1_norm_all, l2_norm_all, all_norm_all, save_fig=False):
    epochs = epoch_list
    
    # L1范数趋势
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 18))
    fig.suptitle(
        f'L1 Norm Trend (X Noise={noise_intensity_x}, Y Noise={noise_intensity_y})',
        fontsize=20, y=0.98
    )
    
    # Weight L1
    weight_l1 = [l1_norm_all[e]['weight']['total'] for e in epochs]
    ax1.plot(epochs, weight_l1, 'o-', color='#050df7', linewidth=2, markersize=8, label='Weight Total L1')
    for e, val in zip(epochs, weight_l1):
        ax1.text(e, val, f'{val:.2f}', fontsize=10, ha='center', va='bottom')
    ax1.set_title('Weight Total L1 Norm', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('L1 Norm Value', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Bias L1
    bias_l1 = [l1_norm_all[e]['bias']['total'] for e in epochs]
    ax2.plot(epochs, bias_l1, 's-', color='#06fc0a', linewidth=2, markersize=8, label='Bias Total L1')
    for e, val in zip(epochs, bias_l1):
        ax2.text(e, val, f'{val:.2f}', fontsize=10, ha='center', va='bottom')
    ax2.set_title('Bias Total L1 Norm', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('L1 Norm Value', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(epochs)
    
    # 全参数L1
    all_l1 = [all_norm_all[e]['l1'] for e in epochs]
    ax3.plot(epochs, all_l1, '^-', color='#ff7f0e', linewidth=2, markersize=8, label='All Params L1')
    for e, val in zip(epochs, all_l1):
        ax3.text(e, val, f'{val:.2f}', fontsize=10, ha='center', va='bottom')
    ax3.set_title('All Parameters (Weight+Bias) Total L1 Norm', fontsize=16)
    ax3.set_xlabel('Epoch', fontsize=14)
    ax3.set_ylabel('L1 Norm Value', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(alpha=0.3)
    ax3.set_xticks(epochs)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_fig:
        os.makedirs(pic_save_dir, exist_ok=True)
        plt.savefig(f'{pic_save_dir}/l1_norm_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # L2范数趋势
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 18))
    fig.suptitle(
        f'L2 Norm Trend (X Noise={noise_intensity_x}, Y Noise={noise_intensity_y})',
        fontsize=20, y=0.98
    )
    
    # Weight L2
    weight_l2 = [l2_norm_all[e]['weight']['total'] for e in epochs]
    ax1.plot(epochs, weight_l2, 'o-', color='#fc02ce', linewidth=2, markersize=8, label='Weight Total L2')
    for e, val in zip(epochs, weight_l2):
        ax1.text(e, val, f'{val:.2f}', fontsize=10, ha='center', va='bottom')
    ax1.set_title('Weight Total L2 Norm', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('L2 Norm Value', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Bias L2
    bias_l2 = [l2_norm_all[e]['bias']['total'] for e in epochs]
    ax2.plot(epochs, bias_l2, 's-', color='#04ecf8', linewidth=2, markersize=8, label='Bias Total L2')
    for e, val in zip(epochs, bias_l2):
        ax2.text(e, val, f'{val:.2f}', fontsize=10, ha='center', va='bottom')
    ax2.set_title('Bias Total L2 Norm', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('L2 Norm Value', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(epochs)
    
    # 全参数L2
    all_l2 = [all_norm_all[e]['l2'] for e in epochs]
    ax3.plot(epochs, all_l2, '^-', color='#9400d3', linewidth=2, markersize=8, label='All Params L2')
    for e, val in zip(epochs, all_l2):
        ax3.text(e, val, f'{val:.2f}', fontsize=10, ha='center', va='bottom')
    ax3.set_title('All Parameters (Weight+Bias) Total L2 Norm', fontsize=16)
    ax3.set_xlabel('Epoch', fontsize=14)
    ax3.set_ylabel('L2 Norm Value', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(alpha=0.3)
    ax3.set_xticks(epochs)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_fig:
        os.makedirs(pic_save_dir, exist_ok=True)
        plt.savefig(f'{pic_save_dir}/l2_norm_combined.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===================== 8. 可视化函数3：训练/测试损失曲线 =====================
def plot_train_test_loss(save_fig=False):
    # 加载双噪声命名的损失文件
    loss_history_filename = f'loss_history_x{noise_intensity_x}_y{noise_intensity_y}.npy'
    loss_test_filename = f'loss_test_x{noise_intensity_x}_y{noise_intensity_y}.npy'
    loss_history = np.load(os.path.join(check_name, loss_history_filename))
    loss_test = np.load(os.path.join(check_name, loss_test_filename))
    epochs_all = list(range(1, len(loss_history) + 1))
    
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(epochs_all, loss_history, color='#ff6b6b', linewidth=2, label='Training Loss')
    ax.plot(epochs_all, loss_test, color='#4ecdc4', linewidth=2, label='Testing Loss')
    
    # 标记关键epoch
    for e in epoch_list:
        if e == 0:
            train_loss_val = loss_history[0]
            test_loss_val = loss_test[0]
        elif e < len(loss_history):
            train_loss_val = loss_history[e]
            test_loss_val = loss_test[e]
        else:
            continue
        
        ax.scatter(e, train_loss_val, color='#ff6b6b', s=120, zorder=5, edgecolor='black')
        ax.scatter(e, test_loss_val, color='#4ecdc4', s=120, zorder=5, edgecolor='black')
        ax.text(e, train_loss_val + 0.02, f'{train_loss_val:.4f}', fontsize=10, ha='center', va='bottom')
        ax.text(e, test_loss_val - 0.02, f'{test_loss_val:.4f}', fontsize=10, ha='center', va='top')
    
    ax.set_title(
        f'Training & Testing Loss Curve (X Noise={noise_intensity_x}, Y Noise={noise_intensity_y})',
        fontsize=18
    )
    ax.set_xlabel('Training Epoch', fontsize=14)
    ax.set_ylabel('MSE Loss', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, len(loss_history) + 50)
    ax.set_ylim(0, 1.5)  # 适配y噪声增大后的损失
    
    plt.tight_layout()
    if save_fig:
        os.makedirs(pic_save_dir, exist_ok=True)
        plt.savefig(f'{pic_save_dir}/train_test_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===================== 9. 主流程 =====================
if __name__ == '__main__':
    # 检查训练目录
    if not os.path.exists(check_name):
        raise FileNotFoundError(
            f"文件夹 {check_name} 不存在！\n"
            f"请先运行x+y双噪声的训练代码，确保生成模型和损失文件。"
        )
    
    # 初始化存储
    global_stats_dict = {}
    l1_norm_all = {}
    l2_norm_all = {}
    all_norm_all = {}
    param_count_all = {}
    
    # 遍历epoch分析
    for epoch in epoch_list:
        model_path = os.path.join(check_name, f'model_{epoch}.pth')
        print(f"\n==================== {epoch} Epoch 模型分析（X噪声={noise_intensity_x}, Y噪声={noise_intensity_y}） ====================")
        
        # 提取参数和统计
        param_dict, stats_dict, global_stats, model = extract_param_distribution(model_path)
        global_stats_dict[epoch] = global_stats
        
        # 计算范数
        l1_norm_dict, l2_norm_dict, all_norm_dict = calculate_norm(
            param_dict, global_stats['all_params']['flatten']
        )
        l1_norm_all[epoch] = l1_norm_dict
        l2_norm_all[epoch] = l2_norm_dict
        all_norm_all[epoch] = all_norm_dict
        
        # 计算参数量
        param_count_dict = calculate_param_count(model)
        param_count_all[epoch] = param_count_dict
        
        # 打印结果
        print("\n===== 权重(Weight)分布统计 =====")
        for layer_name, stats in stats_dict['weight'].items():
            print(f"\n层 {layer_name}:")
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")
        print(f"\n全局Weight统计:")
        for key, value in global_stats['weight'].items():
            if key != 'flatten':
                print(f"  {key}: {value:.4f}")
        
        print("\n===== 偏置(Bias)分布统计 =====")
        for layer_name, stats in stats_dict['bias'].items():
            print(f"\n层 {layer_name}:")
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")
        print(f"\n全局Bias统计:")
        for key, value in global_stats['bias'].items():
            if key != 'flatten':
                print(f"  {key}: {value:.4f}")
        
        print("\n===== 全参数(Weight+Bias)分布统计 =====")
        for key, value in global_stats['all_params'].items():
            if key != 'flatten':
                print(f"  {key}: {value:.4f}")
        
        print_l1_norm_results(l1_norm_dict, epoch)
        print_l2_norm_results(l2_norm_dict, epoch)
        print_all_params_norm_results(all_norm_dict, epoch)
        print_param_count(param_count_dict, epoch)
    
    # 生成可视化图表
    print("\n==================== 开始生成可视化图表 ====================")
    plot_param_distribution_combined(global_stats_dict, save_fig=True)
    plot_norm_trend(l1_norm_all, l2_norm_all, all_norm_all, save_fig=True)
    plot_train_test_loss(save_fig=True)
    
    # 汇总信息
    print("\n==================== 参数量汇总 ====================")
    ref_param = param_count_all[epoch_list[0]]
    print(f"模型总参数量: {ref_param['total_params']:,}")
    print(f"可训练参数量: {ref_param['trainable_params']:,}")
    print(f"Weight总参数: {ref_param['weight_total']:,}")
    print(f"Bias总参数: {ref_param['bias_total']:,}")
    print(f"\n可视化结果保存至: {os.path.abspath(pic_save_dir)}")