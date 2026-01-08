'''
提取保存的MLP模型权重/偏置分布 + L1/L2范数分开输出 + 模型参数量统计 + 训练/测试损失可视化
适配场景：仅给x加噪声的MLP训练代码
核心调整：
1. 修复导入/路径/变量名不匹配问题，代码可独立运行；
2. 统一噪声强度变量名（noise_intensity_x），匹配训练代码；
3. 优化可视化标题（明确标注X噪声），修复L2范数子图标题字体大小；
4. 修复epoch=0时损失索引的潜在bug；
5. 补充缺失的导入，优化代码结构。
'''
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.stats import skew, kurtosis, gaussian_kde
from diff_stage_rand_seed_x import noise_intensity_x

# ===================== 配置项（与训练代码严格对齐） =====================
# ！！！需手动设置为训练代码中使用的噪声强度 ！！！
# noise_intensity_x = 0.05  # 训练代码中x的噪声强度（核心：必须和训练代码一致）
check_name = f'./checkpoint_x{str(noise_intensity_x)}/'  # 匹配训练代码的文件夹名
pic_save_dir = f'weight_plots_x{noise_intensity_x}'     # 可视化结果保存目录

# 全局绘图配置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100  # 基础分辨率
plt.rcParams['savefig.dpi'] = 300 # 保存图片分辨率

# 加载epoch列表（从训练代码保存的文件读取）
epoch_list = np.load(os.path.join(check_name, 'epoch_list.npy')).tolist()

# ===================== 1. 定义模型结构（与训练时完全一致） =====================
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

# ===================== 2. 核心函数1：提取权重/偏置并计算分布统计 =====================
def extract_param_distribution(model_path):
    """提取模型参数（weight/bias/全参数）的分布统计"""
    model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
    model.eval()
    
    param_dict = {'weight': {}, 'bias': {}}
    stats_dict = {'weight': {}, 'bias': {}}
    global_stats = {'weight': {}, 'bias': {}}
    all_params_flatten = np.array([])  # 全参数展平数组
    
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
        # 计算单一层参数的统计特征
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
    
    # 计算Weight/Bias全局统计（所有层合并）
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
    
    # 全参数（Weight+Bias）全局统计
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

# ===================== 3. 核心函数2：计算范数（含全参数范数） =====================
def calculate_norm(param_dict, all_params_flatten):
    """计算weight/bias/全参数的L1/L2范数（分层+全局）"""
    l1_norm_dict = {'weight': {}, 'bias': {}}
    l2_norm_dict = {'weight': {}, 'bias': {}}
    
    # 计算weight/bias各自的范数
    for param_type in ['weight', 'bias']:
        l1_layer = {}
        l2_layer = {}
        total_l1_sum = 0.0
        total_l2_sq_sum = 0.0  # L2范数先算平方和，最后开根号
        total_count = 0
        
        for layer_name, param_flatten in param_dict[param_type].items():
            # 分层范数
            l1 = np.sum(np.abs(param_flatten))
            l2 = np.sqrt(np.sum(np.square(param_flatten)))
            l1_layer[layer_name] = l1
            l2_layer[layer_name] = l2
            
            # 累计全局范数
            total_l1_sum += l1
            total_l2_sq_sum += np.sum(np.square(param_flatten))
            total_count += len(param_flatten)
        
        # 全局范数（所有层合并）
        total_l2 = np.sqrt(total_l2_sq_sum)
        avg_l1 = total_l1_sum / total_count if total_count > 0 else 0
        avg_l2 = total_l2 / np.sqrt(total_count) if total_count > 0 else 0
        
        l1_norm_dict[param_type] = l1_layer
        l1_norm_dict[param_type]['total'] = total_l1_sum
        l1_norm_dict[param_type]['avg'] = avg_l1
        
        l2_norm_dict[param_type] = l2_layer
        l2_norm_dict[param_type]['total'] = total_l2
        l2_norm_dict[param_type]['avg'] = avg_l2
    
    # 全参数（Weight+Bias）范数
    all_params_l1 = np.sum(np.abs(all_params_flatten))
    all_params_l2 = np.sqrt(np.sum(np.square(all_params_flatten)))
    all_norm_dict = {'l1': all_params_l1, 'l2': all_params_l2}
    
    return l1_norm_dict, l2_norm_dict, all_norm_dict

# ===================== 4. 核心函数3：计算模型参数量 =====================
def calculate_param_count(model):
    """统计模型各层参数量（weight/bias/总参数）"""
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
        
        # 区分weight和bias
        if 'weight' in name:
            param_count_dict['weight_total'] += param_num
        elif 'bias' in name:
            param_count_dict['bias_total'] += param_num
    
    return param_count_dict

# ===================== 5. 辅助打印函数 =====================
def print_l1_norm_results(l1_norm_dict, epoch):
    """打印weight/bias的L1范数（分层+全局）"""
    print(f"\n===== {epoch} Epoch L1范数结果（Weight/Bias分开） =====")
    for param_type in ['weight', 'bias']:
        print(f"\n【{param_type.capitalize()} L1范数】")
        for layer_name, l1 in l1_norm_dict[param_type].items():
            if layer_name in ['total', 'avg']:
                continue
            print(f"  层 {layer_name}: L1 = {l1:.4f}")
        print(f"  全局(total): L1 = {l1_norm_dict[param_type]['total']:.4f}")
        print(f"  平均(avg): L1 = {l1_norm_dict[param_type]['avg']:.6f}")

def print_l2_norm_results(l2_norm_dict, epoch):
    """打印weight/bias的L2范数（分层+全局）"""
    print(f"\n===== {epoch} Epoch L2范数结果（Weight/Bias分开） =====")
    for param_type in ['weight', 'bias']:
        print(f"\n【{param_type.capitalize()} L2范数】")
        for layer_name, l2 in l2_norm_dict[param_type].items():
            if layer_name in ['total', 'avg']:
                continue
            print(f"  层 {layer_name}: L2 = {l2:.4f}")
        print(f"  全局(total): L2 = {l2_norm_dict[param_type]['total']:.4f}")
        print(f"  平均(avg): L2 = {l2_norm_dict[param_type]['avg']:.6f}")

def print_all_params_norm_results(all_norm_dict, epoch):
    """打印全参数（Weight+Bias）的L1/L2范数"""
    print(f"\n===== {epoch} Epoch 全参数范数结果（Weight+Bias合并） =====")
    print(f"  全参数L1范数（绝对值和）: {all_norm_dict['l1']:.4f}")
    print(f"  全参数L2范数（平方和开根号）: {all_norm_dict['l2']:.4f}")

def print_param_count(param_count_dict, epoch):
    """打印模型参数量统计"""
    print(f"\n===== {epoch} Epoch 模型参数量统计 =====")
    print(f"总参数量: {param_count_dict['total_params']:,}")
    print(f"可训练参数量: {param_count_dict['trainable_params']:,}")
    print(f"Weight总参数: {param_count_dict['weight_total']:,}")
    print(f"Bias总参数: {param_count_dict['bias_total']:,}")
    print("\n【分层参数量】")
    for layer_name, num in param_count_dict['layer_params'].items():
        print(f"  层 {layer_name}: {num:,} 个参数")

# ===================== 6. 可视化函数1：参数分布整合（Weight/Bias/全参数） =====================
def plot_param_distribution_combined(global_stats_dict, save_fig=False):
    """
    整合参数分布可视化：
    - 3行2列子图：每行对应Weight/Bias/全参数，每列对应直方图+箱线图
    """
    # 配色（匹配训练代码的渐变色）
    color_list = [
        "#00d8b6", "#00c4e0", "#00a8ff", "#4080ff", "#7060ff",
        "#9040ff", "#b020ff", "#e040d0", "#ff5080", "#ff8040", "#ffb020"
    ]
    color_map = {epoch: color_list[i] for i, epoch in enumerate(epoch_list)}
    label_map = {epoch: f'{epoch} Epochs' for epoch in epoch_list}
    
    # 创建3行2列子图
    fig, axes = plt.subplots(3, 2, figsize=(30, 24))
    fig.suptitle(f'Parameter Distribution (X Noise Intensity = {noise_intensity_x})', fontsize=22, y=0.98)
    
    # 定义参数类型和标题
    param_types = ['weight', 'bias', 'all_params']
    param_titles = [
        'Weight Parameters', 
        'Bias Parameters', 
        'All Parameters (Weight + Bias)'
    ]
    
    for row_idx, (param_type, title) in enumerate(zip(param_types, param_titles)):
        ax_hist = axes[row_idx, 0]  # 直方图+KDE
        ax_box = axes[row_idx, 1]   # 箱线图
        
        # 绘制每个epoch的直方图+KDE
        for epoch in epoch_list:
            param_flatten = global_stats_dict[epoch][param_type]['flatten']
            color = color_map[epoch]
            label = label_map[epoch]
            
            # 直方图（密度分布）
            ax_hist.hist(param_flatten, bins='auto', density=True, alpha=0.5, color=color, label=label)
            # KDE曲线（平滑拟合）
            x_range = np.linspace(min(param_flatten), max(param_flatten), 200)
            kde = gaussian_kde(param_flatten)
            ax_hist.plot(x_range, kde(x_range), color=color, linewidth=2)
        
        # 直方图美化
        ax_hist.set_title(f'{title} - Histogram + KDE', fontsize=18)
        ax_hist.set_xlabel('Parameter Value', fontsize=14)
        ax_hist.set_ylabel('Density', fontsize=14)
        ax_hist.legend(fontsize=12, loc='upper right')
        ax_hist.grid(alpha=0.3)
        
        # 绘制每个epoch的箱线图
        box_data = [global_stats_dict[epoch][param_type]['flatten'] for epoch in epoch_list]
        box_labels = [label_map[epoch] for epoch in epoch_list]
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # 箱线图配色
        for patch, color in zip(bp['boxes'], color_map.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        # 箱线图美化
        ax_box.set_title(f'{title} - Box Plot', fontsize=18)
        ax_box.set_xlabel('Epoch', fontsize=14)
        ax_box.set_ylabel('Parameter Value', fontsize=14)
        ax_box.grid(alpha=0.3)
        ax_box.tick_params(axis='x', rotation=45)  # 旋转x轴标签避免重叠
    
    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出标题空间
    if save_fig:
        os.makedirs(pic_save_dir, exist_ok=True)
        plt.savefig(f'{pic_save_dir}/param_distribution_combined.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===================== 7. 可视化函数2：范数趋势（L1/L2整合） =====================
def plot_norm_trend(l1_norm_all, l2_norm_all, all_norm_all, save_fig=False):
    """绘制L1/L2范数趋势（Weight/Bias/全参数各一个子图）"""
    epochs = epoch_list
    # ========== L1范数趋势 ==========
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 18))
    fig.suptitle(f'L1 Norm Trend (X Noise Intensity = {noise_intensity_x})', fontsize=20, y=0.98)
    
    # 子图1：Weight总L1范数
    weight_l1 = [l1_norm_all[e]['weight']['total'] for e in epochs]
    ax1.plot(epochs, weight_l1, 'o-', color='#050df7', linewidth=2, markersize=8, label='Weight Total L1')
    for e, val in zip(epochs, weight_l1):
        ax1.text(e, val, f'{val:.2f}', fontsize=10, ha='center', va='bottom')
    ax1.set_title('Weight Total L1 Norm', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('L1 Norm Value', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(epochs)  # 仅显示关键epoch
    
    # 子图2：Bias总L1范数
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
    
    # 子图3：全参数总L1范数
    all_l1 = [all_norm_all[e]['l1'] for e in epochs]
    ax3.plot(epochs, all_l1, '^-', color='#ff7f0e', linewidth=2, markersize=8, label='All Params (Weight+Bias) L1')
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
    
    # ========== L2范数趋势 ==========
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 18))
    fig.suptitle(f'L2 Norm Trend (X Noise Intensity = {noise_intensity_x})', fontsize=20, y=0.98)
    
    # 子图1：Weight总L2范数
    weight_l2 = [l2_norm_all[e]['weight']['total'] for e in epochs]
    ax1.plot(epochs, weight_l2, 'o-', color='#fc02ce', linewidth=2, markersize=8, label='Weight Total L2')
    for e, val in zip(epochs, weight_l2):
        ax1.text(e, val, f'{val:.2f}', fontsize=10, ha='center', va='bottom')
    ax1.set_title('Weight Total L2 Norm', fontsize=16)  # 修复原代码字体大小不一致
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('L2 Norm Value', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(epochs)
    
    # 子图2：Bias总L2范数
    bias_l2 = [l2_norm_all[e]['bias']['total'] for e in epochs]
    ax2.plot(epochs, bias_l2, 's-', color='#04ecf8', linewidth=2, markersize=8, label='Bias Total L2')
    for e, val in zip(epochs, bias_l2):
        ax2.text(e, val, f'{val:.2f}', fontsize=10, ha='center', va='bottom')
    ax2.set_title('Bias Total L2 Norm', fontsize=16)  # 修复原代码字体大小不一致
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('L2 Norm Value', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(epochs)
    
    # 子图3：全参数总L2范数
    all_l2 = [all_norm_all[e]['l2'] for e in epochs]
    ax3.plot(epochs, all_l2, '^-', color='#9400d3', linewidth=2, markersize=8, label='All Params (Weight+Bias) L2')
    for e, val in zip(epochs, all_l2):
        ax3.text(e, val, f'{val:.2f}', fontsize=10, ha='center', va='bottom')
    ax3.set_title('All Parameters (Weight+Bias) Total L2 Norm', fontsize=16)  # 修复原代码字体大小不一致
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
    """绘制训练/测试损失曲线，标记关键epoch"""
    # 加载损失数据（匹配训练代码的文件名）
    loss_history = np.load(os.path.join(check_name, f'loss_history_{noise_intensity_x}.npy'))
    loss_test = np.load(os.path.join(check_name, f'loss_test_{noise_intensity_x}.npy'))
    epochs_all = list(range(1, len(loss_history) + 1))
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # 绘制损失曲线
    ax.plot(epochs_all, loss_history, color='#ff6b6b', linewidth=2, label='Training Loss')
    ax.plot(epochs_all, loss_test, color='#4ecdc4', linewidth=2, label='Testing Loss')
    
    # 标记关键epoch（修复epoch=0的索引bug）
    for e in epoch_list:
        if e == 0:
            # epoch=0对应loss_history[0]（训练前）
            train_loss_val = loss_history[0]
            test_loss_val = loss_test[0]
        elif e < len(loss_history):
            train_loss_val = loss_history[e]
            test_loss_val = loss_test[e]
        else:
            continue  # 超出范围则跳过
        
        # 绘制标记点
        ax.scatter(e, train_loss_val, color='#ff6b6b', s=120, zorder=5, edgecolor='black')
        ax.scatter(e, test_loss_val, color='#4ecdc4', s=120, zorder=5, edgecolor='black')
        
        # 添加数值标签
        ax.text(e, train_loss_val + 0.01, f'{train_loss_val:.4f}', fontsize=10, ha='center', va='bottom')
        ax.text(e, test_loss_val - 0.01, f'{test_loss_val:.4f}', fontsize=10, ha='center', va='top')
    
    # 图表美化
    ax.set_title(f'Training & Testing Loss Curve (X Noise Intensity = {noise_intensity_x})', fontsize=18)
    ax.set_xlabel('Training Epoch', fontsize=14)
    ax.set_ylabel('MSE Loss', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, len(loss_history) + 50)
    ax.set_ylim(0, 0.7)
    
    # 保存图片
    plt.tight_layout()
    if save_fig:
        os.makedirs(pic_save_dir, exist_ok=True)
        plt.savefig(f'{pic_save_dir}/train_test_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===================== 9. 主流程 =====================
if __name__ == '__main__':
    # 检查训练数据目录是否存在
    if not os.path.exists(check_name):
        raise FileNotFoundError(
            f"文件夹 {check_name} 不存在！\n"
            f"请先运行训练代码，确保生成该目录及模型文件。"
        )
    
    # 初始化存储字典
    global_stats_dict = {}    # 各epoch参数分布统计
    l1_norm_all = {}          # 各epoch L1范数
    l2_norm_all = {}          # 各epoch L2范数
    all_norm_all = {}         # 各epoch 全参数范数
    param_count_all = {}      # 各epoch 参数量
    
    # 遍历每个关键epoch，分析模型
    for epoch in epoch_list:
        model_path = os.path.join(check_name, f'model_{epoch}.pth')
        print(f"\n==================== {epoch} Epoch 模型分析 ====================")
        
        # 1. 提取参数和分布统计
        param_dict, stats_dict, global_stats, model = extract_param_distribution(model_path)
        global_stats_dict[epoch] = global_stats
        
        # 2. 计算范数
        l1_norm_dict, l2_norm_dict, all_norm_dict = calculate_norm(
            param_dict, 
            global_stats['all_params']['flatten']
        )
        l1_norm_all[epoch] = l1_norm_dict
        l2_norm_all[epoch] = l2_norm_dict
        all_norm_all[epoch] = all_norm_dict
        
        # 3. 计算参数量
        param_count_dict = calculate_param_count(model)
        param_count_all[epoch] = param_count_dict
        
        # 4. 打印统计结果
        # 4.1 权重/偏置/全参数分布统计
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
        
        # 4.2 范数结果
        print_l1_norm_results(l1_norm_dict, epoch)
        print_l2_norm_results(l2_norm_dict, epoch)
        print_all_params_norm_results(all_norm_dict, epoch)
        
        # 4.3 参数量统计
        print_param_count(param_count_dict, epoch)
    
    # ===================== 可视化所有结果 =====================
    print("\n==================== 开始生成可视化图表 ====================")
    # 1. 参数分布整合图
    plot_param_distribution_combined(global_stats_dict, save_fig=True)
    
    # 2. 范数趋势图
    plot_norm_trend(l1_norm_all, l2_norm_all, all_norm_all, save_fig=True)
    
    # 3. 训练/测试损失曲线
    plot_train_test_loss(save_fig=True)
    
    # ===================== 最终汇总 =====================
    print("\n==================== 所有Epoch参数量汇总 ====================")
    ref_param = param_count_all[epoch_list[0]]
    print(f"模型总参数量: {ref_param['total_params']:,}")
    print(f"可训练参数量: {ref_param['trainable_params']:,}")
    print(f"Weight总参数: {ref_param['weight_total']:,}")
    print(f"Bias总参数: {ref_param['bias_total']:,}")
    print(f"\n所有可视化结果已保存至：{os.path.abspath(pic_save_dir)}")