import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F, Parameter

import commons
from LayerNorm import LayerNorm


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
    self.encoder = RelativePositionTransformer(
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


class RelativePositionTransformer(nn.Module):
  def __init__(
      self,
      hidden_channels: int,
      filter_channels: int,
      n_heads: int,
      n_layers: int,
      kernel_size: int = 1,
      p_dropout: float = 0.,
      window_size: int = 4,
      **kwargs
  ):
    """
    :param hidden_channels: 隐藏层的维度
    :param filter_channels: Feedforward层中的卷积层的输出维度
    :param n_heads: MultiHeadAttention中的头数
    :param n_layers: Encoder中的层数
    :param kernel_size: Feedforward层中卷积核的大小
    :param p_dropout: 自我注意力和Feed-Forward内部层的dropout率
    :param window_size: 关系注意力窗口大小。如果为4，则对于每个时间步，下一个和前一个4个时间步会被关注。如果使用None，则禁用相对编码，模型变为普通Transformer。
    """

    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size

    self.drop = nn.Dropout(p_dropout)
    # 用于保存多个模型实例，在模型中进行参数共享和参数复用
    # MultiHeadAttention
    self.attn_layers = nn.ModuleList()
    # LayerNorm
    self.norm_layers_1 = nn.ModuleList()
    # Feedforward
    self.ffn_layers = nn.ModuleList()
    # LayerNorm
    self.norm_layers_2 = nn.ModuleList()

    # 对每一层进行实例化
    for i in range(self.n_layers):
      self.attn_layers.append(
        MultiHeadAttention(
          hidden_channels,
          hidden_channels,
          n_heads,
          p_dropout=p_dropout,
          window_size=window_size
        )
      )
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(
        FeedForwardNetwork(
          hidden_channels,
          hidden_channels,
          filter_channels,
          kernel_size,
          p_dropout=p_dropout
        )
      )
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    """
    :param x: 输入序列 (batch_size, seq_len, hidden_channels)
    :param x_mask: 输入序列的掩码，用于将填充部分的信息屏蔽掉 (batch_size, seq_len)
    """

    # 计算一个attention掩码
    # x_mask在第二个维度上unsqueeze（扩展）两次，然后进行逐元素相乘。
    # 这样得到的掩码可以用于在计算MultiHeadAttention中屏蔽掉无效的位置。
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    # 屏蔽掉填充的信息
    x = x * x_mask

    for i in range(self.n_layers):
      # 使用MultiHeadAttention模型self.attn_layers[i]，对输入x进行attention计算，计算出attention的结果y；
      y = self.attn_layers[i](x, x, attn_mask)
      # 对attention结果y进行dropout
      y = self.drop(y)
      # 将x和y的结果进行残差连接，即x + y，然后使用LayerNorm模型self.norm_layers_1[i]对结果进行标准化，得到编码后的输出x；
      x = self.norm_layers_1[i](x + y)

      # 使用Feedforward模型self.ffn_layers[i]对编码后的输出x进行前向传播计算，计算出结果y；
      y = self.ffn_layers[i](x, x_mask)
      # 对y进行dropout
      y = self.drop(y)
      # 再次将x和y的结果进行残差连接，对结果进行标准化
      x = self.norm_layers_2[i](x + y)

    # 将编码后的输出x与x_mask进行逐元素相乘，以屏蔽掉填充的信息
    x = x * x_mask
    return x


class MultiHeadAttention(nn.Module):
  """
  多头相对位置编码注意力模型。

  它学习一个窗口邻居的位置嵌入。对于键和值，它学习不同的嵌入向量集。
  键的嵌入向量通过注意力得分聚合，而值的嵌入向量则通过输出聚合。

  相对注意力窗口大小为2的示例：

  - input = [a，b，c，d，e]
  - rel_attn_embeddings = [e(t-2)，e(t-1)，e(t+1)，e(t+2)]

  因此，它单独为键和值向量学习了4个嵌入向量（总共8个）。

  对于输入c：

  - e(t-2) 对应于 c -> a
  - e(t-2) 对应于 c -> b
  - e(t-2) 对应于 c -> d
  - e(t-2) 对应于 c -> e

  这些嵌入向量在不同的时间步骤之间共享。因此，输入a，b，d和e也使用相同的嵌入。

  当相对窗口超出第一个和最后n项的限制时，将忽略嵌入。
  """

  def __init__(
      self,
      channels: int,
      out_channels: int,
      n_heads: int,
      p_dropout: float = 0.,
      window_size: Optional[int] = None,
      heads_share=True,
      block_length: Optional[int] = None,
      proximal_bias=False,
      proximal_init=False
  ):
    """
    :param channels: 输入的通道数
    :param out_channels: 输出的通道数
    :param n_heads: 注意力头数
    :param p_dropout: dropout 概率
    :param window_size: 关系注意力窗口大小。如果为4，则对于每个时间步，将关注前面和后面的4个时间步
    :param heads_share: 是否共享注意力头
    :param block_length: 位置编码的输入长度
    :param proximal_bias: 是否使用近端偏差
    :param proximal_init: 是否使用近端初始化，将键和查询层的权重初始化为相同
    """

    super().__init__()
    assert channels % n_heads == 0, " [!] channels should be divisible by num_heads."

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init
    self.attn = None

    # 将输入的通道数除以头的数量，得到每个头应该有的通道数。
    self.k_channels = channels // n_heads
    # query, key, value layers
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    # output layers
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    # 在模型训练时随机将一些神经元的输出设置为零，以防止过拟合
    self.drop = nn.Dropout(p_dropout)

    # relative positional encoding layers
    # 如果指定了窗口大小
    if window_size is not None:
      # 如果 heads_share 为 True，则所有头将共享相同的相对位置嵌入；
      # 否则，每个头都会有自己的相对位置嵌入。
      # n_heads_rel 变量表示头的数量。
      n_heads_rel = 1 if heads_share else n_heads
      # 计算相对位置嵌入的标准差，它取决于头的数量和通道数。
      rel_stddev = self.k_channels ** -0.5
      # 创建一个形状为 (n_heads_rel, window_size * 2 + 1, self.k_channels) 的张量，
      # 表示相对位置嵌入的键（key）的权重。
      # 张量的值是从均值为零、标准差为 rel_stddev 的正态分布中随机生成的。
      # nn.Parameter 用于将张量标记为模型的参数，以便在模型训练过程中进行优化。
      self.emb_rel_k = nn.Parameter(
        torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
      )
      # 值（value）的权重
      self.emb_rel_v = nn.Parameter(
        torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
      )

    # 使用 Xavier 均匀初始化方法初始化权重
    nn.init.xavier_uniform_(self.conv_q.weight)
    # 这种初始化方法会将权重值从一个均匀分布中随机采样
    nn.init.xavier_uniform_(self.conv_k.weight)
    # 使其在前向传播和反向传播过程中保持方差不变
    nn.init.xavier_uniform_(self.conv_v.weight)

    if proximal_init:
      # 避免梯度的计算
      with torch.no_grad():
        # 将查询部分的权重和偏置复制给键部分的权重和偏置，从而使它们在初始化时具有相同的值
        # 这种方法可以增强多头自注意力层的性能
        self.conv_k.weight.copy_(self.conv_q.weight)
        self.conv_k.bias.copy_(self.conv_q.bias)  # type: ignore

  def forward(self, x: Tensor, c: Tensor, attn_mask: Optional[Tensor] = None):
    """
    :param x: 输入张量 :math:`[B, C, T]`
    :param c: 用于计算注意力的上下文张量 :math:`[B, C, T]`
    :param attn_mask: 遮盖张量 :math:`[B, 1, T, T]`
    """

    q = self.conv_q(x)  # 查询张量
    k = self.conv_k(c)  # 键张量
    v = self.conv_v(c)  # 值张量

    x, self.attn = self.attention(q, k, v, mask=attn_mask)
    x = self.conv_o(x)

    return x

  def attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    # b为batch size，
    # d为每个输入样本的embedding size，
    # t_s为source sequence length，
    # t_t为target sequence length，
    # n_h为head的个数，
    # d_k为每个head的embedding size。

    # b, d, t_s, t_t = (*key.size(), query.size(2))
    b, d, t_s = key.size()  # type: int, int, int
    t_t: int = query.size(2)

    # 输入的 query, key, value 进行了形状转换和维度交换，将输入的形状从
    # [batch_size, sequence_length, hidden_size] 转换成
    # [batch_size, num_heads, key_channels, sequence_length]
    # 这样的转换有利于并行计算多头自注意力，可以将多个不同的头分别作为不同的通道来处理，加速计算。

    # 每个位置的向量会被拆成 num_heads 份，每份 k_channels 个元素。
    # 这里使用 view 方法将形状转换，并使用 transpose 方法将最后两维交换位置，使得最后一维变成 sequence_length，方便之后计算注意力。
    # [b,h,c,t_t]
    query = query.view(
      b,
      self.n_heads,
      self.k_channels,
      t_t
    )
    # [b,h,c,t_s]
    key = key.view(
      b,
      self.n_heads,
      self.k_channels,
      t_s
    )
    # [b,h,c,t_s]
    value = value.view(
      b,
      self.n_heads,
      self.k_channels,
      t_s
    )

    # 多头注意力机制中的得分计算部分
    # 将query与key进行点积计算，并除以一个归一化因子 $\sqrt{d_k}$ (是为了防止点积过大或过小导致梯度消失或爆炸)，得到得分矩阵 scores
    # [b,h,t_t,t_s]
    scores = torch.einsum('bhdt,bhds -> bhts', query / math.sqrt(self.k_channels), key)

    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."

      # key 的相对位置编码
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      # 相对位置得分矩阵
      # [b,h,c,t_t],[h or 1,e,c] -> [b,h,t_t,e]
      rel_logits = torch.einsum(
        'bhdt,hed->bhte',
        query / math.sqrt(self.k_channels), key_relative_embeddings
      )
      # 绝对位置得分矩阵
      scores_local = self._relative_position_to_absolute_position(rel_logits)
      scores = scores + scores_local

    if self.proximal_bias:
      # 加上一个"邻近偏置"，这种偏置用于鼓励模型在计算注意力时更加依赖先前的位置
      # 当处理自注意力时（即 t_s 等于 t_t），邻近偏置才可用
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)

    if mask is not None:
      # 将分数张量 scores 中需要掩盖的位置用一个极小的值 -1e4 进行填充，使得在计算 softmax 时这些位置对应的概率接近于0
      scores = scores.masked_fill(mask == 0, -1e4)

      if self.block_length is not None:
        # 进行局部掩码操作，self.block_length 指定了局部区域的长度
        assert t_s == t_t, "Local attention is only available for self-attention."

        # 构造一个上三角下三角为 1，其余位置为 0 的张量
        block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        # 只让局部区域内的位置参与注意力计算，而忽略其他位置
        scores = scores.masked_fill(block_mask == 0, -1e4)

    # 对得分矩阵进行处理，得到归一化的attention权重矩阵
    # [b, h, t_t, t_s]
    p_attn = F.softmax(scores, dim=-1)
    # 防止过拟合
    p_attn = self.drop(p_attn)
    # 进行点积操作，得到最终的输出结果
    # [b,h,c,t_s],[b,h,t_t,t_s] -> [b,h,c,t_t]
    output = torch.einsum('bhcs,bhts->bhct', value, p_attn)

    if self.window_size is not None:
      # 使用相对位置编码来调整输出结果
      # 转换为相对位置的权重
      # # [b, h, t_t, 2*t_t-1]
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      # 从self.emb_rel_v中获取相对位置编码
      # [h or 1, 2*t_t-1, c]
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      # 将相对位置权重和相对位置编码相乘
      # [b, h, t_t, 2*t_t-1],[h or 1, 2*t_t-1, c] -> [b, h, c, t_t]
      output = output + torch.einsum(
        'bhte,hec->bhct',
        relative_weights, value_relative_embeddings
      )

    # [b, h, c, t_t] -> [b, d, t_t]
    output = output.view(b, d, t_t)
    return output, p_attn

  @staticmethod
  def _matmul_with_relative_values(p_attn: Tensor, re: Tensor):
    """
    将输入序列中所有位置的相对位置编码和特征编码相乘，得到每个位置对其他位置的贡献

    :param p_attn: 注意力权重 :math:`[B, H, T, V]`
    :param re: 相对值嵌入向量 (a_(i,j)^V) :math:`[H or 1, V, D]`
    :return: :math:`[B, H, T, D]`
    """

    logits = torch.matmul(p_attn, re.unsqueeze(0))
    return logits

  @staticmethod
  def _matmul_with_relative_keys(query: Tensor, re: Tensor):
    """
    :param query: 查询向量的批量。(x*W^Q) :math:`[B, H, T, D]`
    :param re: 相对位置键嵌入向量。(a_(i,j)^K) :math:`[H or 1, V, D]`
    :return: :math:`[B, H, T, V]`

    b: batch size
    h: 头数为
    l: 序列长度
    d: 特征维度
    m: 相对位置嵌入矩阵的长度
    """

    # 对于输入的 x 和 y，矩阵乘法的结果是一个形状为 [b, h, l, 1, d] 的张量，
    # 然后使用 unsqueeze(3) 操作将第 3 个维度扩展为 m，
    # 最后使用 transpose(-2, -1) 操作将张量的最后两个维度交换，变成形状为 [b, h, l, d, m]
    logits = torch.einsum('bhld,hmd -> bhlm', query, re)
    return logits

  def _get_relative_embeddings(self, relative_embeddings: Parameter, length: int) -> Tensor:
    """
    将嵌入向量转换为嵌入张量
    :param relative_embeddings: 相对位置编码矩阵
    :param length: 长度
    """

    # Pad first before slice to avoid using cond ops.
    # 用于填充相对位置编码矩阵的长度差值，以使其能够满足给定长度的要求
    pad_length = max(length - (self.window_size + 1), 0)  # type: ignore
    # 子矩阵的起始位置
    slice_start_position = max((self.window_size + 1) - length, 0)  # type: ignore
    # 结束位置
    slice_end_position = slice_start_position + 2 * length - 1

    if pad_length > 0:
      # 在矩阵的第二维（也就是相对位置的维度）上进行填充
      # 将前面填充 pad_length 个位置，后面也填充 pad_length 个位置
      padded_relative_embeddings = F.pad(
        relative_embeddings,
        commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]])
      )
    else:
      padded_relative_embeddings = relative_embeddings

    # 从填充后的相对位置编码矩阵中切片
    # （batch_size，2*length-1，embed_dim）
    used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]

    return used_relative_embeddings

  @staticmethod
  def _relative_position_to_absolute_position(x: Tensor):
    """
    将相对位置索引的张量转换为本地注意力的绝对位置索引张量。

    :param x: :math:`[B, C, T, 2 * T - 1]`
    :return: :math:`[B, C, T, T]`
    """

    batch, heads, length, _ = x.size()  # type: int, int, int, int
    # 通过在矩阵的右侧添加一列全零的元素，将相对位置得分矩阵x的形状从[b, h, l, 2l-1]扩展为[b, h, l, 2l]
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    # 将x沿第三个维度展平为形状为[b, h, l * 2 * l]的张量x_flat
    x_flat = x.view([batch, heads, length * 2 * length])
    # 在x_flat的右侧添加了一定数量的全零元素，以便x_flat的形状变为[b, h, l * 2 * l + l - 1]。
    # 这里添加的元素数量是l-1，是因为绝对位置得分矩阵的列数比相对位置得分矩阵少了l-1列
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

    # Reshape and slice out the padded elements.
    # 将x_flat重新塑形为[b, h, l+1, 2*l-1]，并取出不包含填充元素的部分，即[:, :, :length, length-1:]
    x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
    return x_final

  @staticmethod
  def _absolute_position_to_relative_position(x):
    """
    将绝对位置编码变换为相对位置编码
    :param x: :math:`[B, C, T, T]`
    :return :math:`[B, C, T, 2*T-1]` 每个元素表示了在序列中两个位置之间的相对距离
    """

    batch, heads, length, _ = x.size()  # type: int, int, int, int
    # padd along column
    # 沿着列方向填充了 length-1 个 0
    # 从 [b, h, l, l] 扩展为 [b, h, l, 2l-1]
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
    # 将 x 展平成形状为 [b, h, l^2+l(l-1)]
    x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
    # add 0's in the beginning that will skew the elements after reshape
    # 沿着行方向再填充 length 个 0
    # 从 [b, h, l^2+l(l-1)] 扩展为 [b, h, l(l+1)]
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    # reshape 回形状为 [b, h, l, 2*l]，然后将其中的第一个元素（也就是“0”位置）舍去
    x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]

    return x_final

  @staticmethod
  def _attention_bias_proximal(length):
    """
    生成一种注意力掩码，鼓励自注意力机制在计算注意力时更注重相邻位置的信息，以抑制远距离的注意力值。
    Bias for self-attention to encourage attention to close positions.

    :param length: an integer scalar.
    :return a Tensor with shape [1, 1, length, length]
    """

    # 生成一个长度为 length 的浮点数序列
    # L
    r = torch.arange(length, dtype=torch.float32)
    # 使用广播运算计算出每个位置与其它位置之间的距离 diff
    # L x L
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    # 并将 diff 的每个元素取绝对值并加 1，再取对数并取相反数，
    # scale mask values
    diff = -torch.log1p(torch.abs(diff))
    # 最后将结果乘以 -1，得到的张量就是正交偏置项
    # 1 x 1 x L x L
    return diff.unsqueeze(0).unsqueeze(0)


class FeedForwardNetwork(nn.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      filter_channels: int,
      kernel_size: int,
      p_dropout: float = 0.,
      activation: Optional[str] = None,
      causal=False
  ):
    """
    带有残差连接的前馈神经网络 Feed-Forward层

    :param in_channels: 输入张量的通道数
    :param out_channels: 输出张量的通道数
    :param filter_channels: 卷积层的过滤器通道数
    :param kernel_size: 卷积核大小
    :param p_dropout: Dropout概率
    :param activation: 激活函数类型 relu或gelu，默认为relu
    :param causal: 进行因果卷积
    """

    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.activation = activation
    self.causal = causal

    # 因果卷积，需要将输入的前面一部分进行 padding，而后面一部分不进行 padding，因为后面一部分的数据是不能被当前的时间步看到的。
    # 如果不是因果卷积，那么将在两侧进行 padding，以保证卷积后输出大小不变
    if causal:
      self.padding = self._causal_padding
    else:
      self.padding = self._same_padding

    # 将输入张量x的通道数从in_channels变为filter_channels
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
    # 将输出张量通道数从filter_channels变为out_channels
    self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
    self.drop = nn.Dropout(p_dropout)

  def forward(self, x: Tensor, x_mask: Tensor) -> Tensor:
    x = self.conv_1(self.padding(x * x_mask))

    # 非线性激活
    # TODO：可以删除 gelu
    if self.activation == "gelu":
      x = x * torch.sigmoid(1.702 * x)
    else:
      x = torch.relu(x)

    x = self.drop(x)
    x = self.conv_2(self.padding(x * x_mask))

    # 掩码张量在这里用于将 padding 的部分的输出置为 0
    return x * x_mask

  def _causal_padding(self, x: Tensor) -> Tensor:
    # 如果卷积核大小为1，则无需进行padding
    if self.kernel_size == 1:
      return x

    # 在时间轴（第三个维度）上
    # 向左（过去）padding kernel_size-1 个位置，
    # 向右（未来）padding 0 个位置，
    # 保证输出的时间轴上的每个位置都依赖于输入的时间轴上的先前位置。
    pad_l = self.kernel_size - 1
    pad_r = 0
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    x = F.pad(x, commons.convert_pad_shape(padding))

    return x

  def _same_padding(self, x: Tensor) -> Tensor:
    if self.kernel_size == 1:
      return x

    pad_l = (self.kernel_size - 1) // 2
    pad_r = self.kernel_size // 2
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    x = F.pad(x, commons.convert_pad_shape(padding))

    return x
