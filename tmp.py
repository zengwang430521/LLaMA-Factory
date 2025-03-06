import torch


def fill_missing_batch(pos, mask):
    """
    pos: torch.LongTensor, shape = (3, batch, n)
    mask: torch.BoolTensor, shape = (batch, n)

    对于 mask 为 False 的位置，所有行都使用 pos 第一行计算出的值进行填充，
    填充值为：pos[0, batch, 左侧最近有效索引] + (当前位置索引 - 左侧最近有效索引)。
    """
    batch, n = mask.shape  # batch 和 n 的大小
    # 生成索引 (batch, n)
    idx = torch.arange(n, device=pos.device).unsqueeze(0).expand(batch, n)

    # 对于 mask=False 的位置，将索引置为 -1
    valid_idx = torch.where(mask, idx, torch.full_like(idx, -1))

    # 计算沿 n 维的累计最大值，得到每个位置左侧最近的有效索引
    cum_max, _ = torch.cummax(valid_idx, dim=1)

    # 计算当前位置与左侧最近有效位置之间的差值
    diff = idx - cum_max

    # 从 pos 第一行中取出有效值，并加上 diff 得到填充值
    # pos[0] 的 shape 为 (batch, n)
    filled = pos[0].gather(dim=1, index=cum_max) + diff

    # 扩展 mask 到 shape (3, batch, n)
    mask_expanded = mask.unsqueeze(0).expand_as(pos)
    # 扩展 filled 到 shape (3, batch, n)
    filled_expanded = filled.unsqueeze(0).expand_as(pos)

    # mask 为 True 的位置保留原 pos 值，否则用 filled 替换
    result = torch.where(mask_expanded, pos, filled_expanded)
    return result

pos = torch.tensor([
    [[1, 2, 2, 0, 0, 3, 4], [10, 11,  0, 0, 11, 15, 16]],  # 第一层 (参考层)
    [[5, 6, 6, 0, 0, 8, 9], [20, 21, 0, 0, 21, 25, 26]],  # 第二层
    [[10, 11, 11, 0, 0, 13, 14], [30, 31,  0, 0, 31, 35, 36]]  # 第三层
])

mask = torch.tensor([
    [True, True, True, False, False, True, True],  # batch=0 的 mask
    [True, True, False, False, True, True, True]   # batch=1 的 mask（不同的情况）
])

output = fill_missing_batch(pos, mask)
print(output)

[[1, 1, 2, 2, 2, 0]]
[[1, 1, 2, 2, 2, 0]]
[[1, 1, 2, 2, 2, 0]]
[[1, 1, 2, 2, 2, 0]]
[[1, 1, 2, 2, 2, 0]]
[[1, 1, 2, 2, 2, 0]]
