import matplotlib.pyplot as plt
import os
import numpy as np

# 创建结果目录（保持原有逻辑）
def create_result_dir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

# 损失曲线绘图（终极修正：适配 1.15.0 版本 LossHistory 正确属性+维度）
def plot_loss_curve(loss_history, exp_name, task_dir):
    save_dir = create_result_dir(f"results/{task_dir}/{exp_name}")
    
    plt.figure(figsize=(10, 6))
    # 1. 正确属性名：loss_train / loss_test（1.15.0 版本专属）
    # 2. 处理二维列表：提取第 0 列（总损失），转为 numpy 数组方便绘图
    train_loss = np.array(loss_history.loss_train)[:, 0]
    test_loss = np.array(loss_history.loss_test)[:, 0]
    
    # 绘制损失曲线（正常显示）
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title(f"Loss Convergence - {exp_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

# 预测对比绘图（保持双种群区分逻辑，无额外修改）
def plot_prediction_vs_true(x, y_true, y_pred, exp_name, task_dir, y_label="Value", is_two_species=False):
    save_dir = create_result_dir(f"results/{task_dir}/{exp_name}")
    plt.figure(figsize=(10, 6))
    
    # 场景 1：Task1 双种群（Lotka-Volterra，猎物+捕食者）
    if is_two_species and y_true.shape[1] == 2 and y_pred.shape[1] == 2:
        # 绘制 猎物（Prey）：第 0 列数据
        plt.plot(x, y_true[:, 0], label="True Prey (猎物)", color="blue", linewidth=2)
        plt.plot(x, y_pred[:, 0], label="Predicted Prey (猎物)", color="red", linewidth=2, linestyle="--")
        # 绘制 捕食者（Predator）：第 1 列数据
        plt.plot(x, y_true[:, 1], label="True Predator (捕食者)", color="orange", linewidth=2)
        plt.plot(x, y_pred[:, 1], label="Predicted Predator (捕食者)", color="green", linewidth=2, linestyle="--")
        # 适配种群主题的标注
        y_label = "Population Number"
    
    # 场景 2：Task2 单函数（导数→原函数，保持原有逻辑）
    else:
        plt.plot(x, y_true, label="True Value", linewidth=2)
        plt.plot(x, y_pred, label="Predicted Value", linewidth=2, linestyle="--")
    
    # 统一格式（兼容报告）
    plt.xlabel("Time (for Task1) / Input (for Task2)")
    plt.ylabel(y_label)
    plt.title(f"Prediction vs True - {exp_name}")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/prediction.png", dpi=300, bbox_inches="tight")
    plt.close()