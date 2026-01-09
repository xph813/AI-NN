import numpy as np

# 5组不同复杂度导数函数与对应原函数
def get_func_set(func_id):
    if func_id == 1:
        return lambda x: x**3, lambda x: 3*x**2  # 简单多项式
    elif func_id == 2:
        return lambda x: -np.cos(2*np.pi*x)/(2*np.pi), lambda x: np.sin(2*np.pi*x)  # 三角函数
    elif func_id == 3:
        return lambda x: np.exp(x) + 0.5*x**2, lambda x: np.exp(x) + x  # 指数+多项式
    elif func_id == 4:
        return lambda x: 0.5*x*np.abs(x), lambda x: np.abs(x)  # 非光滑函数
    elif func_id == 5:
        return lambda x: -np.cos(2*np.pi*x)/(2*np.pi) + np.exp(x), lambda x: np.sin(2*np.pi*x) + np.exp(x)  # 高复杂度复合函数