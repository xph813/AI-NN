import numpy as np
import deepxde as dde
import torch

def lotka_volterra(x, y, noise_level=0.0):
    a, b, c, d = 1, 1, 1, 1
    x_prey, y_predator = y[:, 0:1], y[:, 1:2]
    x_prey_t = dde.grad.jacobian(y, x, i=0)
    y_predator_t = dde.grad.jacobian(y, x, i=1)
    
    eq1 = x_prey_t - a * x_prey + b * x_prey * y_predator
    eq2 = y_predator_t + c * y_predator - d * x_prey * y_predator
    
    if noise_level > 0:
        eq1 += noise_level * torch.randn_like(eq1)  # torch.randn_like 生成与 eq1 同形状、同设备的张量
        eq2 += noise_level * torch.randn_like(eq2)  # 替代 np.random.randn，避免设备不兼容
    return [eq1, eq2]

def lotka_initial_condition(x, noise_level=0.0):
    x0 = 2 + noise_level * np.random.randn()
    y0 = 1 + noise_level * np.random.randn()
    return np.array([x0, y0])