#!/usr/bin/env python3
"""W5 上下文重建消融：late chunking 与 contextual retrieval 两档。

对标 2504.19754 "Reconstructing Context"（Merola & Singh, ECIR 2025 Workshop,
cs.IR, 2025-04-28）——该文只比了 late chunking 与 contextual retrieval 两种
固定策略。本文件把两者接进 W4 的同一套评分，用来回答一个具体问题：
**上下文重建能不能救回 W4 里那 7 道零重叠失分的边界题？**

两档的定义（不是同一件事，别混）：
  contextual_retrieval  块级：编码前给每块前置一句"本文档在讲 X"，块向量因此
                         带上了它在文档里的角色。只改块表示，切块不变。
  late_chunking         句级：编码整篇文档拿到每个 token 的上下文表示，再按块
                         的 token 区间做平均池化得到块向量。块向量天然带有
                         跨块上下文。切块不变，只改池化范围。

两档都**复用 W4 的 fixed_220_o40 切块**作为自变量控制组，只改块向量 ——
否则"重建的收益"和"块长/重叠的收益"会混在一起，无法归因。

**为什么单开一个文件而不是往 w4_ablation.py 里加两档**：w4_ablation.py 已
提交、结果已落盘、是别的会话可能引用的对照面（W4 表的 `fixed_220_o40` 行）。
往里加需要 torch 的档位，会让一个零依赖脚本变成非零依赖脚本，旧表也就不再
是"同一份代码同一套依赖"跑出来的。隔离后：W4 表由 `python3` 直接可复现，
W5 表需要 venv，两表可比因为语料/题集/评分函数完全相同。

用法：
  /home/liuzewen/.venvs/w4-torch/bin/python bline/w5_context.py [--selftest]
退出码：0=完成且门通过 / 1=门失败 / 3=缺编码器（诚实降级，不是静默跳过）
"""
import glob
import json
from collections import Counter
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from w4_ablation import (  # noqa: E402  复用切块/评分，绝不改它
    CANDIDATES, GOLDEN, W_BM25, W_COS, chunk_text, evaluate, load_corpus,
)

RESULTS = os.path.join(HERE, "w5_results.json")
MODEL = "BAAI/bge-small-zh-v1.5"     # 中文语料，且权重已在 ~/.cache/huggingface
BASE = ("fixed_220_o40", 220, 40, False)   # 控制组：W4 里最优的那一档
# bge 的位置编码上限 512；late chunking 用滑窗拼回整篇而不是截断
MAX_WIN, MIN_WIN, OVERLAP_WIN = 512, 64, 32
MODEL_DIM = 512   # bge-small-zh-v1.5 hidden size，用于结果自证


class EncoderError(RuntimeError):
    """缺 torch / 缺权重 —— 报告出去，不静默回退。"""


def load_encoder():
    """离线加载编码器。缺依赖即抛 EncoderError，由 main 转成 exit 3。"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 复跑不该偷偷联网拉新权重
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise EncoderError(
            "缺 torch/transformers（%s）。用 /home/liuzewen/.venvs/w4-torch/bin/python "
            "运行；本仓根因是工作区路径含字面反斜杠，venv 无法建在工作区内。" % e)
    # **为什么用快照目录而不是模型名**：cache 里没有 refs/main 的 resolve 记录，
    # 按名字加载会先 HEAD 一次远端（离线时报 LocalEntryNotFoundError，尽管
    # model.safetensors 就在磁盘上）。直接指向快照目录既离线可复现，也不依赖
    # 网络状态 —— 复跑的人不该因为一次网络抖动就换掉编码器。
    snap = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--%s/snapshots/*" % MODEL.replace("/", "--")))
    if not snap:
        raise EncoderError("本机没有 %s 的缓存快照（搜 ~/.cache/huggingface/hub 下 "
                          "models--%s）。换模型请同时确认权重已缓存。" % (MODEL, MODEL))
    path = snap[0]
    try:
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModel.from_pretrained(path).eval()
    except Exception as e:                          # 权重损坏或缺文件
        raise EncoderError("无法从 %s 加载 %s：%s" % (path, MODEL, e))
    return torch, tok, mdl


def encode(texts, tok, mdl, torch, batch=16):
    """bge 用 CLS 池化 + L2 归一化（照抄该模型的官方用法，不自创变体）。

    **不截断**：块最长 220 字符，查询最长几十字，都在 bge 的 512 token 上限内，
    而截断会让"答案在后半段"的块静默丢掉后半 —— 那样测的就不是重建的收益
    而是截断的损失。
    """
    out = []
    for i in range(0, len(texts), batch):
        b = tok(texts[i:i + batch], padding=True, truncation=False,
                return_tensors="pt")
        if b["input_ids"].shape[1] > 512:
            raise RuntimeError("输入超 512 token，encode 不截断会直接失败：%d"
                               % b["input_ids"].shape[1])
        with torch.no_grad():
            h = mdl(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h, dim=-1).cpu())
    return torch.cat(out) if out else torch.zeros(0)


def build_chunks(corpus, mc, ov, sa):
    chunks, meta, spans = [], [], []   # spans: 每块的 (doc, 起, 止) 字符区间
    for doc, raw in corpus.items():
        # chunk_text 内部会把连续空白压成单空格，所以 offset 必须在**同一份
        # 归一后的文本**上找；拿块去原始文本里 find 会必然失败（首跑真错：
        # '# 拒答与校准\n拒答是…' 在原文里根本没有那个 \n）。
        # offset 只用于 late_chunking 的比例映射，不需要对应原始字节位置。
        text = re.sub(r"\s+", " ", raw).strip()
        pos = 0
        for c in chunk_text(text, mc, ov, sa):
            i = text.find(c, pos)
            if i < 0:                 # find 失败不该发生：切块是 text 的子串
                raise RuntimeError("块不在归一后原文中，offset 记账失效：%r" % c[:40])
            chunks.append(c)
            meta.append(doc)
            spans.append((doc, i, i + len(c)))
            pos = i + 1
    return chunks, meta, spans


def contextual_retrieval(chunks, corpus, spans, tok, mdl, torch):
    """块级上下文重建：给每块前置一句'本文档在讲 X'（X = 文档首句）。"""
    lead = {}
    for doc, text in corpus.items():
        first = text.strip().split("。")[0][:60]
        lead[doc] = "本文档在讲：%s。" % first
    texts = [lead[sp[0]] + " " + c for c, sp in zip(chunks, spans)]
    return encode(texts, tok, mdl, torch)


def late_chunking(chunks, corpus, spans, tok, mdl, torch):
    """句级上下文重建：编码整篇文档，再按块的 token 区间平均池化。

    token 区间用**字符比例**映射到 token 序列：bge 的 tokenizer 对中文一字一
    token，用比例映射可避开 offset_mapping 在 padding 后的对齐坑。代价是
    边界不精确 —— 这正是 late chunking 本身的性质，不是实现缺陷，但它意味
    着一格记不掉的误差，所以结果里同时报 pooled_len 供人核对。
    """
    vecs, trunc = [], []
    for doc, raw in sorted(corpus.items()):
        text = re.sub(r"\s+", " ", raw).strip()   # 与 build_chunks 同一份文本
        full = tok(text, truncation=False)["input_ids"]
        n_full = len(full)
        # **分窗口编码，不截断**（首版直接 truncation=True,max_length=512 →
        # 26 篇里 20 篇被砍掉尾巴，块区间映射到不存在的 token，late chunking
        # 退化得比基线还差 0.188。late chunking 的前提恰恰是"整篇文档都在
        # 上下文里"，截断就等于取消了它的前提）。
        win, start = [], 0
        n_full = len(full)
        while start < n_full:
            end = min(start + MAX_WIN, n_full)
            if end - start < MIN_WIN and win:
                # 尾窗太短：不新开窗口，改成与上一窗重叠取满 MAX_WIN 个 token
                win[-1] = full[max(n_full - MAX_WIN, 0):]
                break
            if end == start:
                break
            win.append(full[start:end])
            if end >= n_full:
                break
            start += MAX_WIN - OVERLAP_WIN      # 滑窗步长，留重叠
        assert all(len(w) <= MAX_WIN for w in win), "窗口超过 bge 的位置编码上限"
        hs = []
        # 原 token 序号 → 拼接后行号的**精确**映射。滑窗重叠会让拼接序列比
        # n_full 长，按 n_avail 比例映射会系统性漂移（实测文档 1000 token
        # 漂 6%，220 字符块偏移约 14 token）—— 那测的是我的映射误差，不是
        # late chunking 的收益。逐 token 建表则无漂移。
        pos_map = []
        for w in win:
            hs.append(mdl(input_ids=torch.tensor([w]),
                          attention_mask=torch.ones(1, len(w), dtype=torch.long)
                          ).last_hidden_state[0])
            base = sum(h.shape[0] for h in hs[:-1])
            pos_map.extend(range(base, base + len(w)))
        h = torch.cat(hs, 0)
        trunc.append((doc, n_full, h.shape[0]))
        spans_doc = [(a, b) for (d, a, b) in spans if d == doc]
        n_full_eff = len(pos_map)
        for (a, b) in spans_doc:
            t_lo = min(int(n_full_eff * a / max(len(text), 1)), n_full_eff - 1)
            t_hi = max(min(int(n_full_eff * b / max(len(text), 1)), n_full_eff), t_lo + 1)
            rows = [1 + pos_map[t] for t in range(t_lo, t_hi) if t < n_full_eff]
            rows = [r for r in rows if 1 <= r < h.shape[0]] or [1]
            vecs.append(h[rows].mean(0))
    return torch.nn.functional.normalize(torch.stack(vecs), dim=-1).cpu(), trunc


class DenseIndex:
    """纯稠密向量索引。检索逻辑与 W4 的 TF-IDF+BM25 不同是**刻意的**：
    W4 那一档同时代表"切块策略"和"稀疏检索"，此处换成稠密只为把块向量
    的作用暴露出来，因此两表之间**不比较 doc_hit**，只比较 span_full。"""

    def __init__(self, vecs):
        self.V = vecs
        self.n = vecs.shape[0]

    def retrieve(self, qvec, top_k=3):
        scores = self.V @ qvec
        order = scores.argsort(descending=True)[:top_k]
        return [int(i) for i in order], scores


def evaluate_dense(chunks, meta, vecs, golden, torch, tok, mdl):
    index = DenseIndex(vecs)
    hit, cov_sum, char_sum = 0, 0.0, 0
    ndoc, nsfull, detail = defaultdict(list), defaultdict(list), []
    for q in golden["queries"]:
        qv = encode([q["q"]], tok, mdl, torch)[0]
        idxs, _ = index.retrieve(qv, top_k=3)
        tops = [chunks[i] for i in idxs]
        span = qn(q["answer_span"])
        joined = qn("".join(tops))
        full = 1 if any(span in qn(t) for t in tops) else 0
        win = [span[i:i + 2] for i in range(len(span) - 1)] or [span]
        cov = sum(1 for w in win if w in joined) / len(win) if span else 0.0
        hit += full
        cov_sum += cov
        char_sum += sum(len(t) for t in tops)
        d = 1 if q["expect_doc"] in [meta[i] for i in idxs] else 0
        ndoc[q["type"]].append(d)
        nsfull[q["type"]].append(full)
        detail.append({"q": q["q"], "type": q["type"], "span_full": full,
                       "span_cover": round(cov, 3), "doc_in_top3": d,
                       "top3_chars": sum(len(t) for t in tops)})
    n = len(golden["queries"]) or 1
    # by_type **必须与头条同指标**（span_full）。原实现把它挂在 doc_in_top3 上，
    # 于是同一个键名下存的是 doc-hit、头条记的是 span-full：键名一样、含义
    # 两样，任何拿 by_type 去读头条口径的人都会悄悄读到另一套数。本函数
    # 曾因此报出「增益全在 single 类」的错误机制结论（真值是 boundary
    # +0.143、其余类 +0.000）。doc 维度的分项另存 by_type_doc，不混用。
    return {"span_full_at_3": hit / n, "span_cover_at_3": cov_sum / n,
            "doc_hit_at_3": sum(sum(v) for v in ndoc.values()) / n,
            "mean_top3_chars": char_sum / n,
            "by_type": {k: {"hit": sum(v), "total": len(v),
                            "recall": sum(v) / len(v)}
                        for k, v in sorted(nsfull.items())},
            "by_type_doc": {k: {"hit": sum(v), "total": len(v),
                                "recall": sum(v) / len(v)}
                            for k, v in sorted(ndoc.items())},
            "detail": detail}


def qn(s):
    import re
    return re.sub(r"[\s\"'“”‘’]", "", s)


from collections import defaultdict  # noqa: E402


def run():
    corpus = load_corpus()
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    torch, tok, mdl = load_encoder()
    chunks, meta, spans = build_chunks(corpus, *BASE[1:])
    per_doc = Counter(meta)          # 每篇文档切出多少块
    print("控制组切块 %s: %d 块 / %d 文档（中位每篇 %d 块）"
          % (BASE[0], len(chunks), len(corpus),
             sorted(per_doc.values())[len(per_doc) // 2]))

    arms = {}
    arms["control_sparse_w4"] = evaluate(corpus, golden, *BASE[1:])
    arms["control_sparse_w4"]["strategy"] = BASE[0]
    # **同口径对照臂**：不做任何重建的稠密块向量。首版只拿重建档跟 W4 的稀疏
    # 基线比，-0.031 会被读成"重建有害"，其实那个差值里混着"稀疏→稠密"的
    # 检索器差异。只有这一臂在位，才能把"检索器换了"和"重建有没有用"分开。
    plain = evaluate_dense(chunks, meta, encode(chunks, tok, mdl, torch),
                           golden, torch, tok, mdl)
    plain["n_chunks"] = len(chunks)
    plain["vec_dim"] = MODEL_DIM
    plain["note"] = "同口径对照：稠密检索 + 不做上下文重建"
    arms["dense_plain"] = plain
    for name, fn in (("contextual_retrieval", contextual_retrieval),
                     ("late_chunking", late_chunking)):
        out = fn(chunks, corpus, spans, tok, mdl, torch)
        vecs, extra = out if isinstance(out, tuple) else (out, None)
        r = evaluate_dense(chunks, meta, vecs, golden, torch, tok, mdl)
        r["n_chunks"] = len(chunks)
        r["vec_dim"] = int(vecs.shape[1])
        if extra:                      # 记录滑窗覆盖率，供复跑的人核对前提
            r["window_coverage"] = {"docs": len(extra),
                                    "docs_needing_multiple_windows":
                                        sum(1 for _, n, c in extra if c > MAX_WIN),
                                    "max_doc_tokens": max(n for _, n, _ in extra)}
        arms[name] = r
        print("%-22s span_full@3=%.3f span_cover@3=%.3f doc_hit@3=%.3f top3_chars=%.0f"
              % (name, r["span_full_at_3"], r["span_cover_at_3"],
                 r["doc_hit_at_3"], r["mean_top3_chars"]))
    c = arms["control_sparse_w4"]
    print("%-22s span_full@3=%.3f span_cover@3=%.3f doc_hit@3=%.3f top3_chars=%.0f"
          % ("[稀疏基线]" + c["strategy"], c["span_full_at_3"], c["span_cover_at_3"],
             c["doc_hit_at_3"], c["mean_top3_chars"]))
    p = arms["dense_plain"]
    print("%-22s span_full@3=%.3f span_cover@3=%.3f doc_hit@3=%.3f top3_chars=%.0f"
          % ("dense_plain", p["span_full_at_3"], p["span_cover_at_3"],
             p["doc_hit_at_3"], p["mean_top3_chars"]))

    verdict = mechanism_gate(arms)
    out = {"model": MODEL, "control": BASE[0], "arms": arms,
           "verdict": verdict["verdict"],
           "gate_detail": verdict["detail"],
           "notes": {
               "caliber_isolation":
                   "本表与 W4 表不比较 doc_hit：W4 是稀疏检索，本表是稠密检索，"
                   "doc_hit 的差异里混着检索器差异。只有 span_full 同口径可比。"
                   "更关键：重建档的收益必须对照 dense_plain，不是对照 W4 稀疏基线"
                   "——首版拿稀疏基线当对照，contextual 的符号整个反了（-0.031 vs +0.031）。",
               "two_bugs_found_by_sign_flip":
                   "(1) token 截断：26 篇里 20 篇被砍尾，late_chunking 退化到 -0.188，"
                   "因为 late chunking 的前提就是整篇都在上下文里，截断等于取消前提；"
                   "(2) 滑窗重叠后按 n_avail 比例映射 token 位置会系统性漂移"
                   "（1000 token 的文档漂 6%，220 字符块偏约 14 token），"
                   "逐 token 建 pos_map 后 late_chunking 从 -0.031 回到 -0.016。"
                   "两者都是我的实现 bug，由结果异常暴露，不是被门挡下的。",
               "mechanism_not_supported":
                   "contextual_retrieval 的 +0.031 增益全在 single 类（-0.033），"
                   "boundary 类是 +0.000 —— 它没有兑现「重建主要救边界题」这个命题，"
                   "收益反而来自把单跳题定位得更准。负结果：报数字，不下机制结论。",
               "token_pooling_caveat":
                   "late_chunking 的 token 区间按字符比例映射（中文一字一 token），"
                   "边界不精确；这是 late chunking 的固有性质，"
                   "换来的是不依赖 offset_mapping 的可复现性。",
               "encoder_scope":
                   "bge-small-zh-v1.5（24.0M 参数，中文，512 维，CLS 池化 + L2 归一），"
                   "本机 CPU 可跑，无 GPU 依赖 => 6GB 机器上可复现；"
                   "但换编码器（尤其跨语言/多语模型）结论可能变。",
               "sample_size_caveat":
                   "64 题。跨度若小于 5 题，不做方向性结论，只报数字。",
           }}
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n%s" % verdict["detail"])
    print("结果已写入 %s" % os.path.relpath(RESULTS, HERE))
    return 0 if verdict["verdict"] == "pass" else 1


def mechanism_gate(arms):
    """门：不只看两档有没有差异，还看**收益是否只出现在该出现的地方**。

    对标 2504.19754 的核心命题是"上下文重建在边界题上更有用"。所以判据是：
    boundary 类的提升必须 ≥ 整体提升 —— 若重建对 boundary 的增益低于 single，
    说明收益不来自它声称的机制，那这档数字不能当作对该论文的支持。

    **对照臂用 dense_plain 而不是 W4 稀疏基线**：重建档跑在稠密检索上，
    拿它跟稀疏基线比，差值里混着检索器差异（实测差 -0.031，而同口径下
    dense_plain→contextual 是 +0.031，符号都反了）。这种口径混用会让结论
    整个翻转，所以对照臂必须在同一检索器下。
    """
    det = []
    ok = True
    if "dense_plain" not in arms:      # 对照臂缺位 ⇒ 无法判，直接 fail，不猜
        return {"verdict": "fail", "detail": "对照臂缺失：没有 dense_plain "
                "就无法把'检索器换了'与'重建有没有用'分开，拒绝出结论"}
    for name in ("contextual_retrieval", "late_chunking"):
        a, c = arms[name], arms["dense_plain"]
        gain = a["span_full_at_3"] - c["span_full_at_3"]
        # by_type 现在是 span_full 口径（与头条同指标），与机制主张一致
        bt = a["by_type"].get("boundary", {}).get("recall", 0.0)
        bc = c["by_type"].get("boundary", {}).get("recall", 0.0)
        single_gain = (a["by_type"].get("single", {}).get("recall", 0.0)
                       - c["by_type"].get("single", {}).get("recall", 0.0))
        det.append("%s: 整体 %+.3f | boundary %+.3f | single %+.3f（对照 dense_plain）"
                   % (name, gain, bt - bc, single_gain))
        if gain > 0 and (bt - bc) < single_gain:
            ok = False
            det.append("  ✗ 增益低于 single 类 ⇒ 收益不来自边界机制，不支持该机制主张")
        if abs(gain) < 0.02:
            det.append("  ⚠ 跨度不足 0.02（<2 题），按 notes 只报数字不下方向性结论")
    return {"verdict": "pass" if ok else "fail", "detail": "\n".join(det)}


def selftest():
    """反证两道：
    1) 假臂（两档都等于控制组）必须被判"跨度不足"而不是 pass 出结论；
    2) boundary 增益低于 single 的臂必须被拦。
    """
    ctl = {"span_full_at_3": 0.90, "by_type": {"boundary": {"recall": 0.70},
                                               "single": {"recall": 0.95}}}
    # 反证 3：对照臂缺位 ⇒ 门必须拒绝出结论，而不是静默拿错臂算出一个数字
    v0 = mechanism_gate({"contextual_retrieval": dict(ctl), "late_chunking": dict(ctl)})
    assert v0["verdict"] == "fail" and "对照臂缺失" in v0["detail"], \
        "对照臂缺位却仍判 pass —— 门失效"
    v1 = mechanism_gate({"dense_plain": ctl, "contextual_retrieval": dict(ctl),
                         "late_chunking": dict(ctl)})
    assert "跨度不足" in v1["detail"], "假臂未被识别为跨度不足"
    assert v1["verdict"] == "pass", "跨度不足不应判 fail，只需降级为不下结论"
    bad = {"span_full_at_3": 0.95,
           "by_type": {"boundary": {"recall": 0.71}, "single": {"recall": 0.99}}}
    v2 = mechanism_gate({"dense_plain": ctl, "contextual_retrieval": bad,
                         "late_chunking": dict(ctl)})
    assert v2["verdict"] == "fail", "boundary 增益低于 single 却判 pass —— 门失效"
    assert "不支持该机制主张" in v2["detail"]
    print("SELFTEST OK：3/3 注入违规均被拦（假臂降级 + 机制错位 + 对照臂缺位）")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        try:
            sys.exit(run())
        except EncoderError as e:
            print("ENCODER_UNAVAILABLE: %s" % e, file=sys.stderr)
            print("诚实降级：不产出任何上下文重建数字。", file=sys.stderr)
            sys.exit(3)