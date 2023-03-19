import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
  def __init__(self, channels: int, eps: float = 1e-5):
    """
    标准的 Layer Normalization 模块，用于对输入进行标准化
    可以有效地缓解神经网络在训练过程中的梯度消失或梯度爆炸问题

    - input: (B, C, T)
    - output: (B, C, T)

    :param channels: 输入的通道数（第二维度）
    :param eps: 用于避免除以 0 的小量
    """

    super().__init__()
    self.channels = channels
    self.eps = eps

    # 模块的 gamma 和 beta 参数通过反向传播来更新
    self.gamma = nn.Parameter(torch.ones(channels))
    self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    # 将输入的特征维度转换到最后一个位置
    x = x.transpose(1, -1)
    # 对输入进行标准化，并使用 gamma 和 beta 参数进行缩放和平移
    x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
    # 将特征维度转换回原来的位置
    return x.transpose(1, -1)
