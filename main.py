import torch
import torch.optim as optim
from FlowMatchingDemo.dataset import get_dataset, get_batch
from model import MLPwithTimeEmbedding
from FlowMatchingDemo.flow_utils import compute_loss
from FlowMatchingDemo.visualization import plot_loss_curve, plot_results, plot_vector_field_evolution
import numpy as np
# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 512
epochs = 30000

# 初始化
data = get_dataset()
print(f"Data mean: {data.mean().item():.4f}, std: {data.std().item():.4f}")
# mean 应接近 0，std 应在0.5到1之间，因为目标数据是外径为1的双环
model = MLPwithTimeEmbedding().to(device)
optimizer = optim.Adam(model.parameters(), lr=4e-3, weight_decay=0.00002)

# 学习率规划: 
#     每过 5000 个 epoch (step_size)
#     学习率乘以 0.5 (gamma)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

loss_history= []
# 训练
print("Starting training...")
for epoch in range(epochs):
    x1 = get_batch(data, batch_size).to(device)
    loss = compute_loss(model, x1)
    curloss = loss.item()   """必须要加.item(), 不然会把整个计算图带进来"""
    loss_history.append(curloss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 500 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
    
plot_loss_curve(loss_history)

plot_results(model, data, device)

plot_vector_field_evolution(model, device)