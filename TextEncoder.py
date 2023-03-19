import math

import torch
from torch import nn, Tensor

import attentions
import commons


class TextEncoder(nn.Module):
  def __init__(
      self,
      n_vocab: int,
      out_channels: int,
      hidden_channels: int,
      filter_channels: int,
      n_heads: int,
      n_layers: int,
      kernel_size: int,
      p_dropout: float
  ):
    """
    将文本序列中的每个单词转换为向量

    :param n_vocab: 嵌入层中字符的数量
    :param out_channels: 输出通道数
    :param hidden_channels: 隐藏通道数
    :param filter_channels: 卷积层的滤波器通道数
    :param n_heads: 注意力头数
    :param n_layers: Transformer层数
    :param kernel_size: Transformer网络中FFN层的卷积核大小
    :param p_dropout: Dropout 概率
    """

    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    # 我们定义了三个神经网络层
    # nn.Embedding 是一个用于将整数编码转换为密集向量表示的层
    # 将输入的离散标记编码成一个隐藏向量表示
    self.emb = nn.Embedding(n_vocab, hidden_channels)
    # 对权重进行初始化，使用正态分布随机初始化权重，均值为0，方差为 hidden_channels ** (幂运算) -0.5。
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

    self.emb_t = nn.Embedding(6, hidden_channels)
    nn.init.normal_(self.emb_t.weight, 0.0, hidden_channels ** -0.5)

    # 多头自注意力机制编码器层
    self.encoder = attentions.RelativePositionTransformer(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout
    )

    # 一维卷积层，hidden_channels 是输入通道数，1 是卷积核的大小
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x: Tensor, t, x_lengths: Tensor):
    """
    :param x: :math:`[B, T]`
    :param x_lengths: :math:`[B]`
    """
    t_zero = (t == 0)
    emb_t = self.emb_t(t)
    emb_t[t_zero, :] = 0

    # 首先对输入 x 进行词嵌入 (self.emb(x))，将单词转换为向量。
    # 然后将其乘以 $\sqrt{hidden_channels}$ 归一化词嵌入向量
    x = (self.emb(x) + emb_t) * math.sqrt(self.hidden_channels)  # [b, t, h]

    # 根据输入 x_lengths 构造掩码矩阵 x_mask，
    # 用于将填充的位置的注意力权重设置为0，避免填充对注意力计算产生影响
    # 形状为 (batch_size, 1, sequence_length)
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(1)), 1).to(x.dtype)  # [b, 1, t]

    # 对 x 进行转置，以适应下面 Encoder 层的输入形状
    # 将维度顺序从 (batch_size, sequence_length, hidden_channels)
    #       转为 (batch_size, hidden_channels, sequence_length)。
    x = torch.einsum('btd,but->bdt', x, x_mask)  # [b, h, t]

    # 对序列进行多头自注意力编码，进行特征提取（残差连接）
    x = self.encoder(x * x_mask, x_mask)

    # 将编码后的序列 x 通过 self.proj 进行投影，特征变换
    # 将 x 的最后一个维度从 hidden_channels 转换为 out_channels * 2
    # 然后将其乘以 x_mask，将填充位置的输出设置为0
    stats = self.proj(x) * x_mask

    # 将 stats 沿着最后一个维度分割为两个部分 高斯分布的均值 m 和 高斯分布的标准差 logs
    # 两个部分是为了用于后续的高斯分布参数计算，用于计算损失函数的均值和标准差
    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask
