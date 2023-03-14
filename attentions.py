import math

import torch
from torch import nn
from torch.nn import functional as F

import commons
from modules import LayerNorm


class Encoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
    """
    :param hidden_channels: 隐藏层的维度
    :param filter_channels: Feedforward层中的卷积层的输出维度
    :param n_heads: MultiHeadAttention中的头数
    :param n_layers: Encoder中的层数
    :param kernel_size: Feedforward层中卷积核的大小
    :param p_dropout: Dropout的概率
    :param window_size: 用于Local Self-Attention的窗口大小
    :param kwargs: 任意数量和类型的关键字
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
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
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


class Decoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
    """
    :param hidden_channels: 隐藏层通道数
    :param filter_channels: 前馈神经网络中卷积层的输出通道数
    :param n_heads: 注意力机制中的头数
    :param n_layers: 解码器中的层数
    :param kernel_size: 前馈神经网络中卷积层的核大小
    :param p_dropout: Dropout 概率
    :param proximal_bias: 是否使用“邻近偏置”
    :param proximal_init: 是否使用“邻近初始化”
    :param kwargs: 其他可选参数
    """

    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init

    self.drop = nn.Dropout(p_dropout)
    self.self_attn_layers = nn.ModuleList()
    self.norm_layers_0 = nn.ModuleList()
    self.encdec_attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()

    # 对每一层进行实例化
    for i in range(self.n_layers):
      self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
      self.norm_layers_0.append(LayerNorm(hidden_channels))
      self.encdec_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask, h, h_mask):
    """
    :param x: decoder input
    :param h: encoder output
    """

    # 生成解码器自注意力层的掩码，采用子序列掩码，将对角线及其右侧的位置设为 0，其余位置设为 1。
    self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    # 生成解码器和编码器之间注意力层的掩码，采用编码器输出和解码器输入的掩码相乘得到，用于过滤掉不需要的信息。
    encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    # 过滤掉不需要的信息。
    x = x * x_mask

    for i in range(self.n_layers):
      # 使用解码器自注意力层计算解码器输入 x 的注意力表示
      y = self.self_attn_layers[i](x, x, self_attn_mask)
      # 应用 dropout
      y = self.drop(y)
      # 归一化
      x = self.norm_layers_0[i](x + y)

      # 使用解码器和编码器之间的注意力层计算编码器输出 h 的注意力表示。
      y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)

      # 使用前馈神经网络计算解码器输入 x 的新表示。
      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)

    x = x * x_mask
    return x


class MultiHeadAttention(nn.Module):
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    """
    :param channels: 输入的通道数
    :param out_channels: 输出的通道数
    :param n_heads: 多头注意力机制中的头数
    :param p_dropout: dropout 概率
    :param window_size: 局部注意力机制中的窗口大小
    :param heads_share: 是否共享多头注意力机制中的参数
    :param block_length: 局部注意力机制中的块长度
    :param proximal_bias: 是否使用邻近偏置
    :param proximal_init: 近似距离匹配
    """

    super().__init__()
    assert channels % n_heads == 0

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
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    # 在模型训练时随机将一些神经元的输出设置为零，以防止过拟合
    self.drop = nn.Dropout(p_dropout)

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
      self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
      # 值（value）的权重
      self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

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
        self.conv_k.bias.copy_(self.conv_q.bias)

  def forward(self, x, c, attn_mask=None):
    """
    :param x: 输入张量
    :param c: 用于计算注意力的上下文张量
    :param attn_mask: 遮盖张量
    """

    q = self.conv_q(x)  # 查询张量
    k = self.conv_k(c)  # 键张量
    v = self.conv_v(c)  # 值张量

    x, self.attn = self.attention(q, k, v, mask=attn_mask)
    x = self.conv_o(x)

    return x

  def attention(self, query, key, value, mask=None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    # b为batch size，
    # d为每个输入样本的embedding size，
    # t_s为source sequence length，
    # t_t为target sequence length，
    # n_h为head的个数，
    # d_k为每个head的embedding size。
    b, d, t_s = key.size()
    t_t = query.size(2)

    # 输入的 query, key, value 进行了形状转换和维度交换，将输入的形状从
    # [batch_size, sequence_length, hidden_size] 转换成
    # [batch_size, num_heads, key_channels, sequence_length]
    # 这样的转换有利于并行计算多头自注意力，可以将多个不同的头分别作为不同的通道来处理，加速计算。

    # 每个位置的向量会被拆成 num_heads 份，每份 k_channels 个元素。
    # 这里使用 view 方法将形状转换，并使用 transpose 方法将最后两维交换位置，使得最后一维变成 sequence_length，方便之后计算注意力。
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    # 多头注意力机制中的得分计算部分
    # 将query与key进行点积计算，并除以一个归一化因子 $\sqrt{d_k}$ (是为了防止点积过大或过小导致梯度消失或爆炸)，得到得分矩阵 scores
    scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."

      # key 的相对位置编码
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      # 相对位置得分矩阵
      rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
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
    p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
    # 防止过拟合
    p_attn = self.drop(p_attn)
    # 进行点积操作，得到最终的输出结果
    output = torch.matmul(p_attn, value)

    if self.window_size is not None:
      # 使用相对位置编码来调整输出结果
      # 转换为相对位置的权重
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      # 从self.emb_rel_v中获取相对位置编码
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      # 将相对位置权重和相对位置编码相乘
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)

    # [b, n_h, t_t, d_k] -> [b, d, t_t]
    output = output.transpose(2, 3).contiguous().view(b, d, t_t)
    return output, p_attn

  def _matmul_with_relative_values(self, x, y):
    """
    将输入序列中所有位置的相对位置编码和特征编码相乘，得到每个位置对其他位置的贡献

    :param x: [b, h, l, m] 当前位置与其他位置的相对位置 (序列中两个位置之间的距离) 编码
    :param y: [h or 1, m, d] 其他位置的特征编码
    :return [b, h, l, d]
    """
    ret = torch.matmul(x, y.unsqueeze(0))
    return ret

  def _matmul_with_relative_keys(self, x, y):
    """
    :param x: [b, h, l, d]
    :param y: [h or 1, m, d]
    :return [b, h, l, m]

    b: batch size
    h: 头数为
    l: 序列长度
    d: 特征维度
    m: 相对位置嵌入矩阵的长度
    """
    # 对于输入的 x 和 y，矩阵乘法的结果是一个形状为 [b, h, l, 1, d] 的张量，
    # 然后使用 unsqueeze(3) 操作将第 3 个维度扩展为 m，
    # 最后使用 transpose(-2, -1) 操作将张量的最后两个维度交换，变成形状为 [b, h, l, d, m]
    ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
    return ret

  def _get_relative_embeddings(self, relative_embeddings, length):
    """
    相对位置编码的子集
    :param relative_embeddings: 相对位置编码矩阵
    :param length: 长度
    """

    # Pad first before slice to avoid using cond ops.
    # 用于填充相对位置编码矩阵的长度差值，以使其能够满足给定长度的要求
    pad_length = max(length - (self.window_size + 1), 0)
    # 子矩阵的起始位置
    slice_start_position = max((self.window_size + 1) - length, 0)
    # 结束位置
    slice_end_position = slice_start_position + 2 * length - 1

    if pad_length > 0:
      # 在矩阵的第二维（也就是相对位置的维度）上进行填充
      # 将前面填充 pad_length 个位置，后面也填充 pad_length 个位置
      padded_relative_embeddings = F.pad(
        relative_embeddings,
        commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    else:
      padded_relative_embeddings = relative_embeddings

    # 从填充后的相对位置编码矩阵中切片
    # （batch_size，2*length-1，embed_dim）
    used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]

    return used_relative_embeddings

  def _relative_position_to_absolute_position(self, x):
    """
    将相对位置得分矩阵转换为绝对位置得分矩阵

    :param x: [b, h, l, 2*l-1]
    :return [b, h, l, l]
    """

    batch, heads, length, _ = x.size()
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

  def _absolute_position_to_relative_position(self, x):
    """
    将绝对位置编码变换为相对位置编码
    :param x: [b, h, l, l]
    :return [b, h, l, 2*l-1] 每个元素表示了在序列中两个位置之间的相对距离
    """

    batch, heads, length, _ = x.size()
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

  def _attention_bias_proximal(self, length):
    """
    生成一个正交偏置项，鼓励自注意力机制在计算注意力时更注重相邻位置的信息
    Bias for self-attention to encourage attention to close positions.

    :param length: an integer scalar.
    :return a Tensor with shape [1, 1, length, length]
    """

    # 生成一个长度为 length 的浮点数序列
    r = torch.arange(length, dtype=torch.float32)
    # 使用广播运算计算出每个位置与其它位置之间的距离 diff
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    # 并将 diff 的每个元素取绝对值并加 1，再取对数并取相反数，
    # 最后将结果乘以 -1，得到的张量就是正交偏置项
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None, causal=False):
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

  def forward(self, x, x_mask):
    x = self.conv_1(self.padding(x * x_mask))

    # 非线性激活
    if self.activation == "gelu":
      x = x * torch.sigmoid(1.702 * x)
    else:
      x = torch.relu(x)

    x = self.drop(x)
    x = self.conv_2(self.padding(x * x_mask))

    # 掩码张量在这里用于将 padding 的部分的输出置为 0
    return x * x_mask

  def _causal_padding(self, x):
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

  def _same_padding(self, x):
    if self.kernel_size == 1:
      return x

    pad_l = (self.kernel_size - 1) // 2
    pad_r = self.kernel_size // 2
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    x = F.pad(x, commons.convert_pad_shape(padding))

    return x
