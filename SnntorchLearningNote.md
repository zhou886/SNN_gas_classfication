# Snntorch学习笔记

[toc]

## 简介

  脉冲神经网络，区别于普通的人工神经网络，它最大的特点在于神经元不再是将所有输入数据加权求和再使用激活函数进行激活，而是采用在离散时间内的将输入的脉冲数据经过加权求和后使用一定的模型输出脉冲数据。

  这种网络的优点在于耗能少且贴近人体大脑的实际工作原理。

  本学习笔记基于snntorch的官方英文教学文档。

## 将输入数据编码成脉冲

  由于脉冲神经网络SNN处理的是脉冲，所以要将采集到的数据统一编码成脉冲，即在离散时间内的0、1串。脉冲编码就是一种将输入数据变为01串的映射方式。

  大体而言，可以将脉冲编码的方式分为以下三种：

+ 速率编码 Rate coding
+ 延迟编码 Latency coding
+ 变化调制 Delta Modulation

### 速率编码

  速率编码的思想是**对于较大的输入则它产生脉冲的概率就越大，反之则越小**。

  对于每个已经正则化的输入数据$X_{ij}$，在每个时间步内它的速率编码为$R_{ij}$。二者服从伯努利分布，即$R_{ij} \sim B(n,p)$，其中$n=1$，生成脉冲的概率$p=X_{ij}$。即有
$$
P(R_{ij}=1)=X_{ij}=1-P(R_{ij}=0)
$$
  在`snntorch`中，使用`spikegen.rate`去生成速率编码。如下所示

```python
from snntorch import spikegen

# Iterate through minibatches
data = iter(train_loader)
data_it, targets_it = next(data)

# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=num_steps)
```

  `spike_data`最后的结构为`[num_steps x batch_size x input dimensions]`。

### 延迟编码

  不同于速率编码在每个时间步内都有可能生成脉冲，延迟编码**在整个时间间隔内只生成一次脉冲**，输入数据越大则脉冲生成越早，反之越迟。

  每个脉冲的生成时间是由RC模型确认的，即有
$$
t_{spike\_time} = \tau[\ln{\frac{X_{ij}}{X_{ij}-threshold}}]
\\
\tau = RC
\tag{1}
$$
<img src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_4_latencyrc.png?raw=true" alt="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_4_latencyrc.png?raw=true" style="zoom:50%;" />

  在`snntorch`中，可以使用`spikegen.latency`来进行延迟编码。如下所示

```python
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)
```

  其中各个超参数的含义如下：

+ tau，RC电路模型的时间常数，tau越大，脉冲生成越迟。
+ threshold，膜电位脉冲触发阈值。
+ linear，将脉冲生成时间的函数简化为线性函数而非对数函数。
+ clip，去掉所有在最后一个时间步才产生的脉冲。

### 变化调制

  很多事实表明神经元是基于变化驱动的，由此引出了变化调制的思想。变化调制只接受**一串根据时间变化**的输入数据，它会比较两个相邻时间步的输入数据，**判断它们之间的差距是否为正且大于阈值，若是则产生一个脉冲**。

  在`snntorch`中，可以使用`spikegen.delta`来进行延迟编码。如下所示

```python
# Convert data
spike_data = spikegen.delta(data, threshold=4)
```

## SNN中的神经元

  在不同的神经网络中有着不同的神经元，在SNN中也有特定的神经元Leaky Integrate-and-fire Neuron(LIF)，列举如下：

+ Lapicque’s RC model: `snntorch.Lapicque`
+ 1st-order model: `snntorch.Leaky`
+ Synaptic Conductance-based neuron model: `snntorch.Synaptic`
+ Recurrent 1st-order model: `snntorch.RLeaky`
+ Recurrent Synaptic Conductance-based neuron model: `snntorch.RSynaptic`
+ Alpha neuron model: `snntorch.Alpha`

### Lapicque's RC model

  将现实中的神经元抽象为一个RC电路模型，就可以得到我们的`Lapicque's RC model`神经元。

<img src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_3_passivemembrane.png?raw=true" alt="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_3_passivemembrane.png?raw=true" style="zoom:50%;" />

#### 参数

+ $I_{in}(t)$，不同时间的输入电流。
+ $U_{mem}$，膜电压。
+ $\tau=RC$，RC模型的时间常数。

<img src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_RCmembrane.png?raw=true" alt="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_RCmembrane.png?raw=true" style="zoom:50%;" />

#### 输入

+ spk_in，输入的脉冲。
+ mem，输入的膜电位。

#### 输出

+ spk_out，下一个时间步的输出脉冲。
+ mem，下一个时间步的膜电位。

#### 使用方法

  在`snntorch`中，使用`snn.Lapicque`来调用`Lapicque's RC model`神经元。如下所示

```python
time_step = 1e-3
R = 5
C = 1e-3

# leaky integrate and fire neuron, tau=5e-3
lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)
```

```python
# A list to store a recording of membrane potential
mem_rec = [mem]

# pass updated value of mem and cur_in[step]=0 at every time step
for step in range(num_steps):
  spk_out, mem = lif1(cur_in[step], mem)

  # Store recordings of membrane potential
  mem_rec.append(mem)

# convert the list of tensors into one tensor
mem_rec = torch.stack(mem_rec)
```

#### 特性

1. 当膜电位超过阈值时，才会有脉冲生成。所以阈值越大，脉冲越难生成，反之则越容易。

	<img src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_spiking.png?raw=true" alt="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_spiking.png?raw=true" style="zoom:50%;" />

2. 在其他条件不变的情况下，输入电流越大，膜电位就能越快到达阈值，从而产生更多的脉冲。

	![https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_reset.png?raw=true](https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_reset.png?raw=true)

	![https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/periodic.png?raw=true](https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/periodic.png?raw=true)

3. 现实中在产生一个脉冲后，神经元会进入一小段时间的休息，在此期间不会产生任何脉冲。该模型使用以下三种方法来实现该机制：

	1. 当膜电位达到或超过阈值，就把膜电位减去阈值大小。`reset_mechanism = "subtract"`这是默认方式。
	2. 当膜电位达到或超过阈值，就直接将膜电位设置为0。`reset_mechanism = "zero"`
	3. 不休息，不做任何处理。

### 1st-order model

  `1st-order model`神经元是对`Lapicque's RC model`神经元的简化。

  在原来的`Lapicque's RC model`神经元中，有以下方程
$$
U(t+\Delta t) = (1-\frac{\Delta t}{\tau})U(t) + \frac{\Delta t}{\tau} I_{\rm in}(t)R \tag{1}
$$

+ 令$R=1$，$\beta = 1 - \frac{\Delta t}{\tau}$。则有
	$$
	U[t+1] = \beta U[t] + (1-\beta)I_{\rm in}[t+1] \tag{2}
	$$

+ 再引入简化的归零休息机制，则有
	$$
	U[t+1] = \underbrace{\beta U[t]}_\text{decay} + \underbrace{WX[t+1]}_\text{input} - \underbrace{S[t]U_{\rm thr}}_\text{reset} \tag{3}
	$$

  使用python实现如下   

```python
def leaky_integrate_and_fire(mem, x, w, beta, threshold=1):
  spk = (mem > threshold) # if membrane exceeds threshold, spk=1, else, 0
  mem = beta * mem + w*x - spk*threshold
  return spk, mem
```

#### 参数

+ $\beta = 1 - \frac{\Delta t}{\tau}$

#### 输入

+ cur_in，输入的电流，是$W \times X[t]$，即将输入脉冲经过一层全连接层。
+ mem，上个时间步的膜电位。

#### 输出

+ spk_out，这个时间步的输出脉冲。
+ mem，这个时间步的膜电位。

#### 使用方法

```python
# Small step current input
w=0.21
cur_in = torch.cat((torch.zeros(10), torch.ones(190)*w), 0)
mem = torch.zeros(1)
spk = torch.zeros(1)
mem_rec = []
spk_rec = []

# neuron simulation
for step in range(num_steps):
  spk, mem = lif1(cur_in[step], mem)
  mem_rec.append(mem)
  spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)
```

### Synaptic Conductance-based neuron model

  `Synaptic Conductance-based neuron model`神经元更加贴近于现实神经元模型，它在简化的`1st-order model`神经元上增加了突触电流的延迟。如下方程所示
$$
\alpha = e^{-\Delta t/\tau_{\rm syn}}
\\
\beta = e^{-\Delta t/\tau_{\rm mem}}
\\
I_{\rm syn}[t+1]=\underbrace{\alpha I_{\rm syn}[t]}_\text{decay} + \underbrace{WX[t+1]}_\text{input}
\\
U[t+1] = \underbrace{\beta U[t]}_\text{decay} + \underbrace{I_{\rm syn}[t+1]}_\text{input} - \underbrace{R[t]}_\text{reset}
\tag{1}
$$

#### 参数

+ $\alpha$，突触电流的延迟参数。
+ $\beta$，膜电位的延迟参数。

#### 输入

+ spk_in，输入的脉冲，即$W X[t]$。
+ syn，在上个时间步的突触电流。
+ mem，在上个时间步的膜电位。

#### 输出

+ spk_out，输出的脉冲。
+ syn，在这个时间步的突触电流。
+ mem，在这个时间步的膜电位。

#### 使用方法

```python
# Temporal dynamics
alpha = 0.9
beta = 0.8
num_steps = 200

# Initialize 2nd-order LIF neuron
lif1 = snn.Synaptic(alpha=alpha, beta=beta)

# Periodic spiking input, spk_in = 0.2 V
w = 0.2
spk_period = torch.cat((torch.ones(1)*w, torch.zeros(9)), 0)
spk_in = spk_period.repeat(20)

# Initialize hidden states and output
syn, mem = lif1.init_synaptic()
spk_out = torch.zeros(1)
syn_rec = []
mem_rec = []
spk_rec = []

# Simulate neurons
for step in range(num_steps):
  spk_out, syn, mem = lif1(spk_in[step], syn, mem)
  spk_rec.append(spk_out)
  syn_rec.append(syn)
  mem_rec.append(mem)

# convert lists to tensors
spk_rec = torch.stack(spk_rec)
syn_rec = torch.stack(syn_rec)
mem_rec = torch.stack(mem_rec)
```

#### 何时使用

+ 输入数据的时间跨度大，没有太多时间上的关系。
+ 输入的脉冲比较稀疏。
+ 在脉冲编码时使用时序编码方式。

  在其他情况下时建议使用`1st-order model`神经元。当`Synaptic`神经元的$\alpha = 0$时，它和`1st-order model`神经元在功能上等价。

  他人经验：在数据集比较简单的情况下，让$\alpha$尽可能小，而当数据集变复杂时，应使用更大的$\alpha$。

## SNN的训练

### THE DEAD NEURON PROBLEM

  在传统的神经网络训练中，梯度下降是最常用的有监督训练算法，即利用计算图来计算各个参数的梯度并通过反向传播来更新参数。但是在脉冲神经网络中，脉冲神经元的脉冲生成里有这样一个方程
$$
\begin{split}S[t] = \begin{cases} 1, &\text{if}~U[t] > U_{\rm thr} \\
0, &\text{otherwise}\end{cases}\end{split}

=\Theta(U[t] - U_{\rm thr})
\tag{1}
$$
<img src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_2_spike_descrip.png?raw=true" alt="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_2_spike_descrip.png?raw=true" style="zoom:50%;" />

  其中的$\Theta(\cdot)$函数的导数在除了$U=\theta$的地方处处为0，在$U=\theta$的地方为$\infin$，这样梯度在反向传播时就会消失或者变成无穷，这就是`dead neuron problem`。

  为了解决这个问题，我们可以用别的梯度来替代$\Theta(\cdot)$函数的梯度，让梯度连续且不让其爆炸或消失。比如在snntorch中默认使用以下梯度来替代
$$
\frac{\partial \tilde{S}}{\partial U} \leftarrow \frac{1}{π}\frac{1}{(1+[Uπ]^2)}
$$
  在snntorch中，你可以使用`snn.LeakySurrogate()`神经元或者`snn.Leaky()`神经元。实际上，在任何神经元中，替代梯度默认总会被调用。如下所示

```python
# Leaky neuron model, overriding the backward pass with a custom function
class LeakySurrogate(nn.Module):
  def __init__(self, beta, threshold=1.0):
      super(LeakySurrogate, self).__init__()

      # initialize decay rate beta and threshold
      self.beta = beta
      self.threshold = threshold
      self.spike_gradient = self.ATan.apply

  # the forward function is called each time we call Leaky
  def forward(self, input_, mem):
    spk = self.spike_gradient((mem-self.threshold))  # call the Heaviside function
    reset = (self.beta * spk * self.threshold).detach()  # remove reset from computational graph
    mem = self.beta * mem + input_ - reset  # Eq (1)
    return spk, mem

  # Forward pass: Heaviside function
  # Backward pass: Override Dirac Delta with the derivative of the ArcTan function
  @staticmethod
  class ATan(torch.autograd.Function):
      @staticmethod
      def forward(ctx, mem):
          spk = (mem > 0).float() # Heaviside on the forward pass: Eq(2)
          ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
          return spk

      @staticmethod
      def backward(ctx, grad_output):
          (spk,) = ctx.saved_tensors  # retrieve the membrane potential
          grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # Eqn 5
          return grad
```

