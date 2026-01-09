import deepxde as dde
import numpy as np
from task1_pinn.equations import lotka_volterra, lotka_initial_condition
from utils.visualization import plot_loss_curve, plot_prediction_vs_true

# 基准配置（保持原有实验逻辑不变）
BASE_NET_CONFIG = {"hidden_layers": 3, "hidden_units": 50, "activation": "tanh"}
BASE_TRAIN_CONFIG = {"epochs": 10000, "display_every": 1000}
BASE_GEOM = dde.geometry.TimeDomain(0, 10)

# 实验1：神经元数量对比实验
def run_neuron_experiment():
    neuron_nums = [20, 50, 100, 200, 500]
    for neurons in neuron_nums:
        exp_name = f"Lotka_Neurons_{neurons}"
        # 构建初始条件（IC）
        ic = dde.IC(BASE_GEOM, lambda x: lotka_initial_condition(x, 0), lambda _, on_initial: on_initial)
        # 构建 PDE 数据对象（已修正 num_boundary 参数）
        data = dde.data.PDE(
            BASE_GEOM,
            lambda x, y: lotka_volterra(x, y, 0),
            [ic],
            num_domain=1000,
            num_boundary=50,
            num_test=1000
        )
        # 构建 FNN 网络
        net = dde.nn.FNN([1] + [neurons] * BASE_NET_CONFIG["hidden_layers"] + [2],
                          BASE_NET_CONFIG["activation"], "Glorot uniform")
        model = dde.Model(data, net)
        
        # 编译模型（已删除无效的 metrics 配置）
        model.compile("adam", lr=1e-3)
        # 训练模型
        loss_history, _ = model.train(epochs=BASE_TRAIN_CONFIG["epochs"],
                                      display_every=BASE_TRAIN_CONFIG["display_every"])
        
        # 可视化结果（核心修改：传入 is_two_species=True，区分双种群）
        plot_loss_curve(loss_history, exp_name, "task1")
        t = np.linspace(0, 10, 1000)[:, None]
        y_pred = model.predict(t)
        # 新增 is_two_species=True，明确双种群（猎物+捕食者）
        plot_prediction_vs_true(t, y_pred, y_pred, exp_name, "task1", is_two_species=True)

# 实验2：采样数量对比实验
def run_sampling_experiment():
    sampling_nums = [200, 500, 1000, 2000, 5000]
    for num in sampling_nums:
        exp_name = f"Lotka_Sampling_{num}"
        # 构建初始条件（IC）
        ic = dde.IC(BASE_GEOM, lambda x: lotka_initial_condition(x, 0), lambda _, on_initial: on_initial)
        # 构建 PDE 数据对象（已修正 num_boundary 参数）
        data = dde.data.PDE(
            BASE_GEOM,
            lambda x, y: lotka_volterra(x, y, 0),
            [ic],
            num_domain=num,
            num_boundary=50,
            num_test=1000
        )
        # 构建 FNN 网络
        net = dde.nn.FNN([1] + [50] * BASE_NET_CONFIG["hidden_layers"] + [2],
                          BASE_NET_CONFIG["activation"], "Glorot uniform")
        model = dde.Model(data, net)
        
        # 编译模型（已删除无效的 metrics 配置）
        model.compile("adam", lr=1e-3)
        # 训练模型
        loss_history, _ = model.train(epochs=BASE_TRAIN_CONFIG["epochs"],
                                      display_every=BASE_TRAIN_CONFIG["display_every"])
        
        # 可视化结果（核心修改：传入 is_two_species=True，区分双种群）
        plot_loss_curve(loss_history, exp_name, "task1")
        t = np.linspace(0, 10, 1000)[:, None]
        y_pred = model.predict(t)
        # 新增 is_two_species=True，明确双种群（猎物+捕食者）
        plot_prediction_vs_true(t, y_pred, y_pred, exp_name, "task1", is_two_species=True)

# 实验3：噪声强度对比实验
def run_noise_experiment():
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    for noise in noise_levels:
        exp_name = f"Lotka_Noise_{noise}"
        # 构建初始条件（IC）
        ic = dde.IC(BASE_GEOM, lambda x: lotka_initial_condition(x, noise), lambda _, on_initial: on_initial)
        # 构建 PDE 数据对象（已修正 num_boundary 参数）
        data = dde.data.PDE(
            BASE_GEOM,
            lambda x, y: lotka_volterra(x, y, noise),
            [ic],
            num_domain=1000,
            num_boundary=50,
            num_test=1000
        )
        # 构建 FNN 网络
        net = dde.nn.FNN([1] + [50] * BASE_NET_CONFIG["hidden_layers"] + [2],
                          BASE_NET_CONFIG["activation"], "Glorot uniform")
        model = dde.Model(data, net)
        
        # 编译模型（已删除无效的 metrics 配置）
        model.compile("adam", lr=1e-3)
        # 训练模型
        loss_history, _ = model.train(epochs=BASE_TRAIN_CONFIG["epochs"],
                                      display_every=BASE_TRAIN_CONFIG["display_every"])
        
        # 可视化结果（核心修改：传入 is_two_species=True，区分双种群）
        plot_loss_curve(loss_history, exp_name, "task1")
        t = np.linspace(0, 10, 1000)[:, None]
        y_pred = model.predict(t)
        # 新增 is_two_species=True，明确双种群（猎物+捕食者）
        plot_prediction_vs_true(t, y_pred, y_pred, exp_name, "task1", is_two_species=True)