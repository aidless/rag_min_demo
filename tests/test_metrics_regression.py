#!/usr/bin/env python3
"""B 线六度量回归门（stdlib-only，fail-closed）。

作用：把 R59/P2-B18 用上游 LongBench 实现验证过的六度量值固化为回归断言。
     任何一次度量实现漂移 → 本门 FAIL（exit≠0）并指认被改坏的那个量。

R59 记档（PHASE_LOG R59）：count 0.5 / retrieval 0.5 / cls 1.0 /
                            f1 0.8182 / qa_f1 0.8 / rouge 0.8
本门的 fixture 逐项复现上述六值（唯一分析脚本=本文件，逐项重算）。

运行：python3 tests/test_metrics_regression.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools.longbench_metrics import (  # noqa: E402
    count_score, retrieval_score, classification_score,
    f1_score, qa_f1_score, rouge_l_score,
)

# ---- R59 六度量目标值（回归基准，来自 PHASE_LOG R59）----
R59 = {
    "count": 0.5,
    "retrieval": 0.5,
    "cls": 1.0,
    "f1": 0.8182,       # 9/11 = 0.818181...
    "qa_f1": 0.8,
    "rouge": 0.8,
}

TOL = 5e-4  # R59 记档为 4 位小数


def approx(a, b, tol=TOL):
    return abs(a - b) <= tol


def main():
    failures = []

    # ---------- 1) 六度量逐项复现 R59 ----------
    fixtures = []

    # count：prediction 含 2 个数字，命中 1 个 → 0.5
    got = count_score("The total is 5 and 6.", "5")
    fixtures.append(("count", got, R59["count"], "5 and 6 vs gt=5"))

    # retrieval：gt 走 'Paragraph N' 契约 → 0.5
    got = retrieval_score("It appears in 3 and 7.", "Paragraph 3")
    fixtures.append(("retrieval", got, R59["retrieval"], "3 and 7 vs Paragraph 3"))

    # classification：唯一命中 → 1.0
    got = classification_score("Sport", "Sport", all_classes=["Sport", "Politics"])
    fixtures.append(("cls", got, R59["cls"], "Sport vs Sport"))

    # f1：11/11 字符袋，公共 9 → 9/11 = 0.8182
    got = f1_score("abcdefghijk", "abcdefghiXY")
    fixtures.append(("f1", got, R59["f1"], "9/11 chars"))

    # qa_f1：5/5 词，公共 4 → 0.8
    got = qa_f1_score("alpha beta gamma delta epsilon",
                      "alpha beta gamma delta zeta")
    fixtures.append(("qa_f1", got, R59["qa_f1"], "4/5 tokens"))

    # rouge-l：10/10 词，LCS 8 → 0.8
    got = rouge_l_score("a b c d e f g h i j", "a b c d e f g h x y")
    fixtures.append(("rouge", got, R59["rouge"], "LCS 8/10"))

    print("[1] R59 六度量回归：")
    for name, actual, expect, note in fixtures:
        ok = approx(actual, expect)
        print(f"   {'PASS' if ok else 'FAIL'}  {name:9s} = {actual:.4f}  "
              f"(期望 {expect:.4f}; {note})")
        if not ok:
            failures.append(f"{name}: {actual:.6f} != {expect}")

    # ---------- 2) 输入契约（R59 两次纠正固化）----------
    print("[2] 输入契约（R59 纠正固化）：")
    for label, fn, args, kwargs in [
        ("retrieval 缺 'Paragraph N'", retrieval_score, ("x", "no anchor"), {}),
        ("cls 缺 all_classes", classification_score, ("x", "y"), {}),
    ]:
        try:
            fn(*args, **kwargs)
            print(f"   FAIL  {label} 未报错（契约失守）")
            failures.append(f"contract not enforced: {label}")
        except (ValueError, KeyError) as e:
            print(f"   PASS  {label} → {type(e).__name__}")
    # 正向对照：给了契约必须成功
    if retrieval_score("in 3", "Paragraph 3") != 1.0:
        failures.append("retrieval 正向对照失败")
        print("   FAIL  retrieval 正向对照")
    else:
        print("   PASS  retrieval 正向对照 = 1.0")

    # ---------- 3) 上游缺陷修正反证 ----------
    # 上游边遍历边 remove 会跳过紧随其后的待删项 → 计数偏大（分数偏小）。
    # 本例：all_classes=["Sport","Spo","Spor"]，pred 全含，gt="Sport"
    #   上游（模拟）：[Sport,Spo,Spor] --删Spo--> [Sport,Spor] --跳过Spor--> 1/2 = 0.5
    #   本实现（正解）：Spo/Spor 皆为 gt 子串噪声 → 仅 [Sport] → 1/1 = 1.0
    print("[3] 上游 classification 缺陷修正反证：")
    got = classification_score("Sport Spo Spor", "Sport",
                               all_classes=["Sport", "Spo", "Spor"])
    if approx(got, 1.0):
        print(f"   PASS  修正实现 = {got:.4f}（正解 1.0）")
    else:
        failures.append(f"cls defect fix: {got} != 1.0")
        print(f"   FAIL  修正实现 = {got:.4f}（应 1.0）")

    def _upstream_mutant(prediction, ground_truth, all_classes):
        """上游原始语义复刻（边遍历边删），仅用于反证对比。"""
        em = [c for c in all_classes if c in prediction]
        for m in em:
            if m in ground_truth and m != ground_truth:
                em.remove(m)
        return (1.0 / len(em)) if ground_truth in em else 0.0

    up = _upstream_mutant("Sport Spo Spor", "Sport", ["Sport", "Spo", "Spor"])
    if approx(up, 0.5):
        print(f"   PASS  上游复刻 = {up:.4f}（证实跳元素缺陷：0.5 ≠ 1.0）")
    else:
        failures.append(f"upstream mutant unexpected: {up}")
        print(f"   FAIL  上游复刻 = {up:.4f}（预期 0.5）")

    # ---------- 4) 非空判据（反证门本身不是恒真）----------
    print("[4] 门自身非空性（注入错误期望必须被抓）：")
    if approx(f1_score("abcdefghijk", "abcdefghiXY"), 0.99):
        failures.append("non-vacuity: 错期望 0.99 未被拦")
        print("   FAIL  错期望 0.99 竟通过（门恒真）")
    else:
        print("   PASS  错期望 0.99 被拦（门非恒真）")

    print()
    if failures:
        print(f"METRICS GATE FAIL: {len(failures)} 项")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("METRICS GATE PASS: R59 六度量 + 契约 + 缺陷修正 + 非空性 全通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
