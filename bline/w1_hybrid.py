"""W1混合检索策略模块（纯函数，不依赖/不修改冻结基线）。

输入为已算好的分数dict {chunk_idx: score}，输出为排序后的 [(score, idx)]。
归一化口径与冻结基线一致：cand内max-norm。
"""
import math


def max_norm(scores):
    m = max(scores.values()) or 1.0
    return {i: s / m for i, s in scores.items()}


def weighted_fusion(cos, bm, w_cos=0.6, candidates=10, top_k=3):
    """余弦初排top-N候选 + max-norm加权融合（基线策略的一般化，基线=w_cos=0.6）。"""
    cand = sorted(cos, key=cos.get, reverse=True)[:candidates]
    cn, bn = max_norm({i: cos[i] for i in cand}), max_norm({i: bm[i] for i in cand})
    fused = [(w_cos * cn[i] + (1.0 - w_cos) * bn[i], i) for i in cand]
    fused.sort(reverse=True)
    return fused[:top_k]


def rrf_fusion(cos, bm, k=60, candidates=10, top_k=3):
    """Reciprocal Rank Fusion：两路排名倒数加权，不依赖分数尺度。"""
    rc = {i: r + 1 for r, i in enumerate(sorted(cos, key=cos.get, reverse=True))}
    rb = {i: r + 1 for r, i in enumerate(sorted(bm, key=bm.get, reverse=True))}
    cand = sorted(cos, key=cos.get, reverse=True)[:candidates]
    fused = [(1.0 / (k + rc[i]) + 1.0 / (k + rb[i]), i) for i in cand]
    fused.sort(reverse=True)
    return fused[:top_k]
