import torch
import torch.utils


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
  """
  用于在多个GPU之间分配batch的数据加载器，会尽量使得每个batch中的样本长度相似，
  通过指定length boundaries来定义长度组，任何一个batch都是包含在两个连续的length boundaries之间的。

  继承自PyTorch的DistributedSampler，它维护了数据集的一些基本信息，例如长度和样本数量，
  构造函数的参数包括数据集、batch size、length boundaries、replicas数量和rank，
  其中replicas数量和rank用于指定当前GPU的数量和ID。

  尽量使得每个batch中的样本长度相似是为了避免在训练过程中产生过多的padding，从而降低训练效率和模型性能。
  在深度学习中，通常会将一个batch中的多个样本同时输入到模型中进行训练，而每个样本的长度不同，
  如果直接将它们拼接在一起形成一个batch进行训练，则需要在长度较短的样本上进行padding，
  使得它们的长度与长度最长的样本相同。这样就会造成一些浪费，
  因为在进行padding时，填充的信息对模型并没有贡献，同时也会增加训练的计算量。
  如果能够使得每个batch中的样本长度相似，则可以减少padding的数量，从而提高训练效率和模型性能。

  TODO：在这里需要修改为自动确定 boundaries

  在一个batch中保持相似的输入长度。
  长度组由边界指定。
  例如）边界 = [b1，b2，b3] -> 任何一个批次都包括{x | b1 < length(x) <= b2}或{x | b2 < length(x) <= b3}。

  它会删除不在边界范围内的样本。
  例如）边界 = [b1，b2，b3] -> 所有满足length(x) <= b1或length(x) > b3的x都会被删除。
  """

  def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
    """
    :param dataset: 数据集 (TextAudioLoader)
    :param batch_size: 每个批次中样本的数量
    :param boundaries: 长度的边界，指定每个批次的长度范围
    :param num_replicas: 进程数，即将数据集分成多少份
    :param rank: 进程的标识符，从0开始
    :param shuffle: 是否在每个epoch开始时打乱数据集
    """

    # 初始化基类
    super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

    self.lengths = dataset.lengths
    self.batch_size = batch_size
    self.boundaries = boundaries

    # 根据边界将数据集分成不同的“桶”，并计算每个“桶”包含的样本数
    self.buckets, self.num_samples_per_bucket = self._create_buckets()
    # 计算整个数据集包含的样本数
    self.total_size = sum(self.num_samples_per_bucket)
    # 每个进程应该处理的样本数
    self.num_samples = self.total_size // self.num_replicas

  def _create_buckets(self):
    """
    将数据集分成不同的桶，每个桶内包含长度在一定范围内的样本
    """

    # 如果边界为[32, 64, 128, 256]，则将创建三个桶：
    # 第一个桶用于包含长度在(32, 64]之间的样本，
    # 第二个桶用于包含长度在(64, 128]之间的样本，
    # 第三个桶用于包含长度在(128, 256]之间的样本。
    buckets = [[] for _ in range(len(self.boundaries) - 1)]

    for i in range(len(self.lengths)):
      length = self.lengths[i]
      # 样本应该被分配到的桶的索引号
      idx_bucket = self._bisect(length)
      # 样本的长度小于最小的范围或大于最大的范围，则返回-1，表示不分配到任何桶中
      if idx_bucket != -1:
        buckets[idx_bucket].append(i)

    # 最后一个bucket开始遍历
    for i in range(len(buckets) - 1, -1, -1):
      if len(buckets[i]) == 0:
        # 移除空桶
        buckets.pop(i)
        # 以便后面的代码可以正确地计算样本的长度范围
        self.boundaries.pop(i + 1)

    num_samples_per_bucket = []
    for i in range(len(buckets)):
      len_bucket = len(buckets[i])
      # 总的 batch size：每个 replica 上的 batch size 乘以 replica 的数量
      total_batch_size = self.num_replicas * self.batch_size
      # 当前 bucket 内需要额外添加多少个样本才能达到一个完整的 batch
      rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
      # 当前 bucket 内需要取 len_bucket + rem 个样本
      num_samples_per_bucket.append(len_bucket + rem)

    # 桶的列表和每个桶中样本的数量列表
    return buckets, num_samples_per_bucket

  def __iter__(self):
    # deterministically shuffle based on epoch
    # 随机数生成器
    g = torch.Generator()
    # 根据固定的种子确定一个确定性随机序列
    g.manual_seed(self.epoch)

    indices = []
    if self.shuffle:
      for bucket in self.buckets:
        # 打乱数据顺序
        indices.append(torch.randperm(len(bucket), generator=g).tolist())
    else:
      for bucket in self.buckets:
        # 每个样本在bucket中的索引
        indices.append(list(range(len(bucket))))

    batches = []
    for i in range(len(self.buckets)):
      bucket = self.buckets[i]
      # 当前bucket中的样本数量
      len_bucket = len(bucket)
      ids_bucket = indices[i]
      # 每个bucket需要的样本数量
      num_samples_bucket = self.num_samples_per_bucket[i]

      # add extra samples to make it evenly divisible
      # 需要添加的额外样本数量rem
      rem = num_samples_bucket - len_bucket
      # 使得每个bucket的样本数都可以被batch size整除
      ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

      # subsample
      # 均匀采样，得到当前进程需要处理的样本id
      ids_bucket = ids_bucket[self.rank::self.num_replicas]

      # batching
      # 将ids_bucket分成若干个batch，每个batch包含batch_size个样本
      for j in range(len(ids_bucket) // self.batch_size):
        batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
        batches.append(batch)

    if self.shuffle:
      # 根据随机数生成的batch_ids列表打乱batches的顺序
      batch_ids = torch.randperm(len(batches), generator=g).tolist()
      batches = [batches[i] for i in batch_ids]
    self.batches = batches

    # 确保生成的所有batch的样本数之和等于num_samples（数据集中的样本总数）
    assert len(self.batches) * self.batch_size == self.num_samples
    # batches的迭代器
    return iter(self.batches)

  def _bisect(self, x, lo=0, hi=None):
    """
    二分查找算法来查找一个数字在已排序的列表中的位置
    :param x: 要查找的数字
    :param lo:
    :param hi: 搜索范围的右端点
    :return:
    """
    if hi is None:
      hi = len(self.boundaries) - 1

    if hi > lo:
      # 中间位置的索引
      mid = (hi + lo) // 2
      if self.boundaries[mid] < x <= self.boundaries[mid + 1]:
        return mid
      elif x <= self.boundaries[mid]:
        # 在搜索范围的左半部分递归
        return self._bisect(x, lo, mid)
      else:
        # 在搜索范围的右半部分递归
        return self._bisect(x, mid + 1, hi)
    else:
      return -1

  def __len__(self):
    # 数据集的长度为多少个batch
    # 如果 num_samples 是 1000，batch_size 是 64，那么将返回 15，表示数据集可以分成 15 个大小为 64 的批次，以处理所有的 1000 个样本
    return self.num_samples // self.batch_size
