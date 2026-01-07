import torch
import torch.nn as nn
import math
    
class MLPwithTimeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # 增加一个时间映射层
        self.time_embed = nn.Sequential(
            nn.Linear(16, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.net = nn.Sequential(
            nn.Linear(2 + 64, 256), # 2D坐标 + 64维时间嵌入
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x, t):
        # 1. 构造高频特征 (Sinusoidal Embedding 简化版)
        freqs = torch.arange(8, device=x.device).float() * math.pi
        t_attr = t * freqs[None, :] # [batch, 8]
        t_emb_raw = torch.cat([torch.sin(t_attr), torch.cos(t_attr)], dim=-1) # [batch, 16]
        
        # 2. 通过一个小 MLP 提取时间特征
        t_features = self.time_embed(t_emb_raw)
        
        # 3. 拼接并预测
        return self.net(torch.cat([x, t_features], dim=-1))