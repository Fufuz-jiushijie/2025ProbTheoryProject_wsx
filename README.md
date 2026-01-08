# 2025秋季概率统计与随机过程大作业
本项目旨在深入浅出地解析生成模型中的 **流匹配（Flow Matching）** 理论及其实现。本项目分为两部分，分别是理论推导与工程示例。

在理论推导部分（本 README 前半段），将展示从随机变量的常微分方程（ODE）出发，如何通过连续性方程（Continuity Equation）构建概率密度的演化路径。  
我们将推导如何利用条件向量场（Conditional Vector Field）的加权平均来构造复杂的边缘向量场（Marginal Vector Field），并从数学上证明为何直接回归条件向量场即可实现对目标分布的拟合。这一部分为 Flow Matching 的底层数学逻辑提供了严谨的支撑。  

在工程示例部分（本 README 后半段），我们将上述理论转化为可执行的代码实现。展示如何构建神经网络 $v_t(x)$ 来学习向量场，并利用数值模拟积分生成样本。  


# 理论推导
本章节深入探讨如何将分布转换问题转化为向量场学习问题。  
主要分为两个核心部分：  
1. 目标构建  
我们从物理学中的质量守恒定律出发，将概率密度演化过程形式化。首先针对每一个样本点 $z$ 定义其独立的常微分方程（ODE）和条件向量场 $u_t(x|z)$ 。通过高斯散度定理，导出描述概率密度随时间变化的偏微分方程。最后利用积分变换，将条件路径合并为边缘概率路径 $p_t(x)$，并建立起边缘向量场与条件概率、数据分布及条件向量场之间的数学关联。  

2. 损失有效性证明  
在建立目标之后，我们将证明按照加权平均构造的边缘向量场，能够严格满足边缘概率路径的连续性方程，确保概率演化的物理自洽性。再通过变分推导展示，在训练神经网络 $v_t(x)$ 时，以条件向量场作为回归目标，恰好等价于难以直接观测的边缘向量场。  

这一结论确立了 Conditional Flow Matching (CFM) 在计算上的简便性与理论上的正确性。
## 1.目标构建
符号说明：
* $z$ ：目标分布  
* $x0$ ：初始化噪声  
* $u_t^{target}(\vec{x}|\vec{z})$ : 条件向量场
* $u_t^{target}(\vec{x})$ ：边缘向量场
* $p_t(x|z)$ ：条件概率路径
* $p_t(x)$ ：边缘概率路径
### Step 1: 随机变量的常微分方程 -> 概率密度的偏微分方程

1. **构建条件 ODE**：
  对于每个固定的辅助随机变量 $\vec{z} = z$，定义条件概率路径下的常微分方程（ODE）：  

   $$\frac{dX_t}{dt} = u_t^{target}(\vec{x}|\vec{z})$$
  其中 $u_t^{target}(\vec{x}|\vec{z})$ 被称为条件向量场。
  

2. **引入质量守恒**：
  考虑空间中体积 $V$ 内的“质量”（即概率密度积分）。根据物理意义，盒内质量的变化率等于流出量的负值：

   $$\frac{dm}{dt} = -\text{流出量}$$


3. **导出条件 PDE**：
 利用高斯散度定理，将通过闭合曲面的流量转化为体积积分：  
   $$\frac{\partial p_t(x|z)}{\partial t} = -\nabla \cdot (p_t(x|z) \cdot u_t^{target}(x|z))$$



   此方程为条件概率密度演化的偏微分方程（PDE）。

---

### Step 2: 用条件概率路径和条件向量场表示边缘概率路径对 $t$ 的导数

1. **边缘化定义**：   
	边缘概率路径 $p_t(x)$ 是联合分布对 $z$ 维度的多重积分：
   $$p_t(x) = \int p_t(x|z) p_{data}(z) dz$$

2. **对时间求导**：   
	由于积分变量 $z$ 与时间 $t$ 无关，可以将求导符号移入积分内：
   $$\frac{\partial p_t(x)}{\partial t} = \int \frac{\partial p_t(x|z)}{\partial t} p_{data}(z) dz$$

3. **代入条件 PDE**：   
	将 Step 1 中的结论代入，得到边缘概率变化率的表达式 ：
   
   $$\frac{\partial p_t(x)}{\partial t} = -\int \nabla \cdot (p_t(x|z) \cdot u_t^{target}(x|z)) p_{data}(z) dz$$

---
### Step 3: 建立边缘向量场与条件概率、pdata、条件向量场的关系
1. **边缘分布的连续性方程**：
	边缘概率路径 $p_t(x)$ 自身也必须满足其对应的连续性方程 ：
   
   $$\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot (p_t(x) \cdot u_t^{target}(x))$$


2. **算子提取**：
	由于散度算子 $\nabla \cdot$ 是对 $x$ 的操作，与积分变量 $z$ 无关，可以将其从积分号中移出：    

   $$\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot \int p_t(x|z) \cdot u_t^{target}(x|z) p_{data}(z) dz$$
  
3. **求解边缘向量场**：  
	对比上述两个方程，为了使等式成立，边缘向量场 $u_t^{target}(x)$ 必须满足：  

   $$u_t^{target}(x) = \frac{\int p_t(x|z) p_{data}(z) u_t^{target}(x|z) dz}{p_t(x)}$$

 **小结**：整个过程利用条件分布构造边缘分布，通过对条件向量场进行加权平均（以 $p_{data}(z)$ 为权重），从而解出满足边缘分布演化的边缘向量场。

## 损失一致性证明
### 定义损失函数  
在条件流匹配中，训练神经网络 $v_t(x)$ 的目标是最小化在所有路径和时间上的平方误差：  

$$\mathcal{L}_{CFM} = \mathbb{E}_{t \sim \mathcal{U}[0,1], z \sim p_{data}(z), x \sim p_t(x|z)} \left[ \| v_t(x) - u_t^{target}(x|z) \|^2 \right]$$




### 利用重期望公式展开
为了分析最优解，我们将期望分解，先对给定 $x$ 时的后验分布 $p_t(z|x)$ 求期望，再对 $x$ 的边缘分布 $p_t(x)$ 求期望：   

$$\mathcal{L}_{CFM} = \mathbb{E}_{t, x \sim p_t(x)} \left[ \mathbb{E}_{z \sim p_t(z|x)} \left[ \| v_t(x) - u_t^{target}(x|z) \|^2 \right] \right]$$



### 求解极值 (MSE 最优解)   
对于每一个固定的 $x$ 和 $t$，神经网络 $v_t(x)$ 要使内部的二次损失最小。根据均方误差（MSE）的性质，最优预测值等于目标变量的**条件期望**：   

$$ v_t^*(x) = \mathbb{E}_{z \sim p_t(z|x)} [ u_t^{target}(x|z) ] $$

### 4. 转化为积分形式并代入贝叶斯公式  
将条件期望展开为关于 $z$ 的积分：   

$$v_t^*(x) = \int u_t^{target}(x|z) p_t(z|x) dz$$

利用贝叶斯公式 $p_t(z|x) = \frac{p_t(x|z)p_{data}(z)}{p_t(x)}$ 代入上式： 

$$v_t^*(x) = \frac{\int p_t(x|z) p_{data}(z) u_t^{target}(x|z) dz}{p_t(x)}$$

### 5. 结论一致性 
观察上述推导结果，发现其形式与第一步Step3最终推出的边缘向量场完全一致：   

$$u_t^{target}(x) = \frac{\int p_t(x|z) p_{data}(z) u_t^{target}(x|z) dz}{p_t(x)}$$

**直观解释：** 由于神经网络在训练时会尝试拟合所有经过 $x$ 点的条件路径的平均速度，这种“多对一”的加权平均在数学上恰好抵消了 $z$ 的依赖项，从而使得回归条件向量场的最优解就是边缘向量场。


# 工程示例

## 项目总体架构：
* dataset.py: 生成 2D 目标分布（Concentric Circles）。  
* model.py: 构建预测向量场的 MLP 网络，集成了多维时间嵌入（Time Embedding）。  
* flow_utils.py: 核心逻辑模块，包含  
1.训练目标：条件流匹配损失 $||v_\theta(x_t, t) - (x_1 - x_0)||^2$  
2.采样器：基于欧拉法（Euler Method）的 ODE 数值求解。
* main.py: 训练循环与超参数控制。
* visualization.py: 不同t的向量场、loss曲线与生成结果的可视化   

以下分模块讲解

## **流匹配损失函数构建**：  
* 考虑到实现简便性，采用直线作为条件概率路径，计算目标向量场。  
* 计算单步模型输出与目标向量场之间的均方误差作为损失函数。

## **模型架构说明**：  
### 1. 时间嵌入层 (Time Embedding Layer)  
为了让网络能够捕捉到 $t \in [0, 1]$ 过程中的微小动力学变化，使用Time Embedding进行升维：  

首先，通过 $t \cdot k\pi$ 将标量 $t$ 映射到 8 个不同的频率空间。利用 $\sin$ 和 $\cos$ 函数生成 16 维的周期性特征向量，克服了神经网络对原始标量输入的谱偏差 (Spectral Bias)。  

其次，使用两层线性层（含 SiLU 激活）将 16 维原始频率特征投影至 64 维的高维嵌入空间。这一步实现了“特征增强”，确保时间信号在后续计算中不会被空间坐标信号淹没。  

### 2. 主干预测网络 (Backbone Network)  
主干网络负责接收融合后的信号并预测当前的漂移速度（Vector Field）：  
* 输入维度: $2 (\text{spatial}) + 64 (\text{temporal}) = 66$ 维。    
* 深度与宽度：采用 4 层 256 维的线性层。  
 初期采用3层MLP，但模型无法捕捉内层圆环特征。增加一层 MLP 是捕捉“内层圆环”的关键。
* 激活函数：全线使用 SiLU (Sigmoid-weighted Linear Unit)。  

**模型数据流**
* 输入:当前粒子位置 $x$ 与时间戳 $t$ 。  
* 升维: $t \xrightarrow{\text{Sin/Cos}} 16d \xrightarrow{\text{Linear}} 64d$ 。  
* 拼接: $[\text{Position}, \text{TimeFeatures}]\in \mathbb{R}^{66}$ 。    
* 映射: $66d \xrightarrow{4\times\text{MLP}}2d$ 。  
* 输出:预测的条件向量场 $u_t^{target}(x)$ ，指导粒子下一步的移动方向。 

## 训练：
核心逻辑：概率路径采样模型并不直接学习终点分布，而是学习将噪声“推”向目标的速度场。在每一个训练迭代中：
* 目标采样 ($x_1$)：从真实数据集中抽取 batch 样本作为流的终点 。  
* 噪声采样 ($x_0$)：从标准正态分布 $\mathcal{N}(0, I)$ 中采样初始点 。
* 路径插值 ($x_t$)：根据线性概率路径 $x_t = (1-t)x_0 + t x_1$ 构造中间状态。
* 损失计算：模型预测 $v_\theta(x_t, t)$，并以理论条件向量场 $u_t^{target} = x_1 - x_0$ 为目标计算均方误差 (MSE) 。

训练代码实现：
```Python
for epoch in range(epochs):
    # 1. 抽取目标数据 batch (x1)
    x1 = get_batch(data, batch_size).to(device)
    
    # 2. 计算流匹配损失
    # 内部包含：采样 t ~ U[0,1], 采样 x0 ~ N(0,I), 构建 xt
    loss = compute_loss(model, x1)
    
    # 3. 显存管理优化
    # 必须使用 .item() 提取标量值，断开计算图引用，防止内存泄漏
    curloss = loss.item() 
    loss_history.append(curloss)
    
    # 4. 反向传播与参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 5. 学习率动态调整
    # 使用 StepLR 策略，配合 epoch 计数自动进行学习率衰减
    scheduler.step()
```
优化器策略：Step Decay学习率调度。采用分段常数衰减（StepLR），每 5000 个 Epoch 将学习率减半（Gamma=0.5）。  

理论依据：在训练初期，较大的学习率有助于模型快速定位向量场的大致径向结构。随着训练深入，减小的学习率使模型能够精细拟合内层圆环的收敛边界，从而解决“模式丢失”问题。  

实验观察：Loss 与 收敛性数值特性：Loss 曲线在降至 0.8 附近时会出现明显的平台期，此时解 ODE 轨迹，粒子已经能清晰地汇聚成两个同心圆环。
![生成结果](./Generation_results.png)

## 推理：
通过数值模拟求解由连续性方程定义的概率流。

## 总结：
本项目通过工程手段验证了推导中的核心结论：通过对条件向量场进行加权平均，可以构造出满足连续性方程的边缘向量场。模型成功地将随机粒子的输运过程从 p0 引导至 p1。