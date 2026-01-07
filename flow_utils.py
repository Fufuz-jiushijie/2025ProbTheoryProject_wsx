import torch

def compute_loss(model, x1):
    batch_size = x1.shape[0]
    # 1. 采样 t ~ U(0, 1)
    t = torch.rand(batch_size, 1).to(x1.device)
    
    # 2. 采样噪声 x0 ~ N(0, I)
    x0 = torch.randn_like(x1)
    
    # 3. 构造条件概率路径 (这里就设置为最简单的直线)
    xt = (1 - t) * x0 + t * x1
    
    # 4. 目标向量场就是直线位移：u_t(x|x0, x1) = x1 - x0
    target = x1 - x0
    
    # 5. 计算网络输出， 计算 MSE
    v_pred = model(xt, t)   # 该点该时间的速度
    loss = torch.mean((v_pred - target)**2) # MSE
    return loss

@torch.no_grad()
def ode_solve(model, x0, steps=1000):
    """Euler Method 求解 ODE"""
    dt = 1.0 / steps
    x = x0.clone()
    for i in range(steps):
        t = torch.ones(x.shape[0], 1).to(x.device) * (i / steps)
        v = model(x, t)
        x = x + v * dt
    return x