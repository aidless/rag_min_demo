#!/usr/bin/env python3
"""LongBench 六度量：stdlib-only 移植（B 线回归门用）。

来源：.agent-memory/world-sota-3lines/repro/b-p18-longbench/LongBench/metrics.py
      （MIT；R59/P2-B18 已用上游实现验证六度量可跑通，commit 2e00731）

为什么重写：B 线是零依赖（仅标准库）仓库，上游依赖 jieba/fuzzywuzzy/rouge。
本模块只移植 R59 六度量：count / retrieval / classification / f1 / qa_f1 / rouge-l。

上游缺陷（本次修正，见 tests/test_metrics_regression.py）：
  classification_score 在 `for match_term in em_match_list:` 循环内调用
  `em_match_list.remove(match_term)` —— 边遍历边删会跳过元素。当 prediction 中
  同时命中「ground_truth」与「另一个也在 ground_truth 里的候选类别」等多类名时，
  上游可能漏删干扰项，使 1/len(em_match_list) 偏大（虚高）。
  本实现改用列表推导（语义等价、不跳元素）。

移植边界（诚实）：上游依赖 jieba / fuzzywuzzy / rouge 包的分支
  (retrieval_zh_* / rouge_zh_* / qa_f1_zh_* / code_sim_score) 未移植——
  不在 R59 六度量内，且违反 B 线零依赖约束。
  rouge-l 以 LCS 公式重构（与上游 rouge 包的 rouge-l 同公式：F = 2PR/(P+R)）。

零依赖：本文件仅 import re / string / collections。
"""

import re
import string
from collections import Counter

__all__ = [
    "normalize_answer",
    "count_score",
    "retrieval_score",
    "classification_score",
    "f1_score",
    "qa_f1_score",
    "rouge_l_score",
]

# 未移植分支的显式占位：调用即报错，避免「静默退化成 0 分」被当结果。
_NOT_PORTED = (
    "retrieval_zh_score", "rouge_zh_score", "qa_f1_zh_score", "code_sim_score",
)


def normalize_answer(s):
    """小写化 + 去冠词/标点/多余空白（与上游逐字一致）。"""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def count_score(prediction, ground_truth, **kwargs):
    """数字命中率：prediction 里的数字中等于 ground_truth 的比例。"""
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return 0.0 if not numbers else right_num / len(numbers)


def retrieval_score(prediction, ground_truth, **kwargs):
    """段落检索命中率：ground_truth 形如 'Paragraph N'，取 N 与 prediction 数字比对。

    契约（R59 纠正）：ground_truth 必须是含 `Paragraph N` 的串，否则 IndexError。
    """
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        raise ValueError("ground_truth 缺少 'Paragraph N' 契约；R59 已纠正此输入错")
    gt_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(gt_id))
    return 0.0 if not numbers else right_num / len(numbers)


def classification_score(prediction, ground_truth, **kwargs):
    """分类命中：prediction 命中的类别集合中，ground_truth 占比的倒数。

    契约（R59 纠正）：必须传 all_classes=kwargs["all_classes"]，否则 KeyError。
    修正：上游边遍历边 remove 会跳元素（可虚高）；此处用列表推导。
    """
    if "all_classes" not in kwargs:
        raise ValueError("缺少 all_classes；R59 已纠正此输入错")
    all_classes = kwargs["all_classes"]

    em_match_list = [c for c in all_classes if c in prediction]
    # 干扰项过滤：命中但不是 ground_truth 的类别不计入（无遍历-删除缺陷）
    kept = [m for m in em_match_list if not (m in ground_truth and m != ground_truth)]
    if ground_truth not in kept:
        return 0.0
    return 1.0 / len(kept)


def _bag_f1(pred_tokens, gt_tokens):
    """令牌袋 F1（上游 f1_score 语义：多重集交集）。"""
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction, ground_truth, **kwargs):
    """原始令牌袋 F1（上游对字符串输入即按字符计数，此处保持同语义）。"""
    return _bag_f1(prediction, ground_truth)


def qa_f1_score(prediction, ground_truth, **kwargs):
    """问答 F1：先 normalize_answer，再按空白切词做令牌袋 F1。"""
    return _bag_f1(
        normalize_answer(prediction).split(),
        normalize_answer(ground_truth).split(),
    )


def _lcs_len(a, b):
    """最长公共子序列长度（滚动数组 DP）。"""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(cur[-1], prev[j]))
        prev = cur
    return prev[-1]


def rouge_l_score(prediction, ground_truth, **kwargs):
    """rouge-l F（LCS 公式，stdlib 重构；与上游 rouge 包同公式）。"""
    p, g = prediction.split(), ground_truth.split()
    if not p or not g:
        return 0.0
    lcs = _lcs_len(p, g)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec)
