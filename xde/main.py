from task1_pinn.experiments import run_neuron_experiment, run_sampling_experiment, run_noise_experiment
from task2_operator.experiments import run_activation_experiment, run_sampling_experiment as run_op_sampling, run_complexity_experiment

import os
# 1. 强制指定 DeepXDE 使用 PyTorch 后端（对 NVIDIA GPU 兼容性最佳，3090 性能拉满）
os.environ["DDE_BACKEND"] = "pytorch"
# 2. 强制指定使用第 0 块 GPU（3090 一般对应 cuda:0，单卡环境直接写 0 即可）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    # 运行Task1所有实验（PINN正问题）
    # print("Running Task 1 Experiments...")
    # run_neuron_experiment()
    # run_sampling_experiment()
    # run_noise_experiment()
    
    # 运行Task2所有实验（算子学习反问题）
    print("Running Task 2 Experiments...")
    run_activation_experiment()
    run_op_sampling()
    run_complexity_experiment()
    
    print("All Experiments Completed! Results are saved in 'results' directory.")