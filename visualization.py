import matplotlib.pyplot as plt
import torch
from FlowMatchingDemo.flow_utils import ode_solve
import numpy as np
def plot_results(model, data, device):
    model.eval()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 0.初始噪声
    x0 = torch.randn(1000, 2).to(device)
    axes[0].scatter(x0[:,0].cpu(), x0[:,1].cpu(), alpha=0.5, s=10, color='blue')
    axes[0].set_title("Original Noise")
    
    # 1.生成结果
    x1_gen = ode_solve(model, x0, steps = 5000)
    axes[1].scatter(x1_gen[:, 0].cpu(), x1_gen[:, 1].cpu(), alpha=0.5, s=10, color='orange')
    axes[1].set_title("Generated Samples")
    
    # 2. 目标分布
    axes[2].scatter(data[:1000, 0], data[:1000, 1], alpha=0.5, s=10)
    axes[2].set_title("Target Data (p1)")
    grid_res = 20
    x_range = torch.linspace(-2, 2, grid_res)
    y_range = torch.linspace(-2, 2, grid_res)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)
    
    plt.savefig('Generation_results.png')
    print("Result saved as Generation_results.png")
    
def plot_vector_field_evolution(model, device):
    times = [0.1, 0.5, 0.9] 
    fig, axes = plt.subplots(1, len(times), figsize=(12, 5))
    
    
    grid_res = 15
    x_range = torch.linspace(-2, 2, grid_res)
    y_range = torch.linspace(-2, 2, grid_res)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)

    for i, t_val in enumerate(times):
        t_vec = torch.ones(grid_points.shape[0], 1).to(device) * t_val
        with torch.no_grad():
            v_field = model(grid_points, t_vec).cpu()
        
        axes[i].quiver(grid_x.flatten(), grid_y.flatten(), v_field[:, 0], v_field[:, 1], color='blue')
        axes[i].set_title(f"Vector Field at t={t_val}")
    
    plt.savefig('vector_fields.png')
    plt.show()
    print("Vector Fields in vector_fields.png")
    
    
    
def plot_loss_curve(loss_history):
    """
    绘制训练 Loss 曲线
    :param loss_history: 包含每轮 loss 值的列表
    """
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color='#2ca02c', label='Flow Matching Loss')
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title('Training Convergence')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    if len(loss_history) > 100:
        smooth_loss = np.convolve(loss_history, np.ones(50)/50, mode='valid')
        plt.plot(smooth_loss, color='#cca02c', lw=2, label='Smoothed Loss')
    plt.legend()
    plt.savefig('loss_curve.png', dpi=300)
    print("loss curve in loss_curve.png")