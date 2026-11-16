#!/usr/bin/env python3
"""Score the frozen hard_v1 set. Implements PROTOCOL.md. Stdlib only.

The question file and protocol were committed before this scorer existed.
This module does not edit them. `--check` recomputes every figure in the
README number block and fails on drift.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag_demo

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.json"
SHA_FILE = HERE / "SHA256.txt"
RESULTS = HERE / "results.json"
README = ROOT / "README.md"

B = 2000
SEED = 20261116
K_LO = (25 * (B - 1)) // 1000
K_HI = (975 * (B - 1) + 999) // 1000
WEIGHTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CHUNKS = [120, 220, 400]
OVERLAP = 40
PRIMARY_CHUNK = 220
PRIMARY_W = 0.6
START = "<!-- hard-v1-numbers:start -->"
END = "<!-- hard-v1-numbers:end -->"
DIGIT_ALLOW = ("python3", "hard_v1", "BM25")


def self_check_rules():
    assert K_LO == 49 and K_HI == 1950 and K_HI < B
    item = {"answerable": True, "expect_docs": ["a"]}
    got = score_item(
        [{"doc_id": "b"}, {"doc_id": "a"}, {"doc_id": "a"}], item
    )
    assert got["hit"] and got["rank"] == 2 and not got["abstain"]
    multi = {"answerable": True, "expect_docs": ["a", "b"]}
    got = score_item(
        [{"doc_id": "a"}, {"doc_id": "a"}, {"doc_id": "b"}], multi
    )
    assert got["hit"] and got["rank"] == 3
    got = score_item([{"doc_id": "a"}] * 3, multi)
    assert not got["hit"] and got["rank"] == 0
    got = score_item([], multi)
    assert got["abstain"] and not got["hit"]
    uns = {"answerable": False, "expect_docs": []}
    got = score_item([{"doc_id": "a"}], uns)
    assert got["hit"] is None and not got["abstain"]
    got = score_item([], uns)
    assert got["abstain"] and got["hit"] is None
    perfect = [(True, 1)] * 8
    ci = bootstrap_cell(perfect)
    assert ci == ((1.0, 1.0), (1.0, 1.0))
    # Identical systems: difference interval collapses to zero, not a win.
    diff_ci = bootstrap_diff(perfect, perfect, "recall")
    assert diff_ci == (0.0, 0.0)
    sample = f"{START}\n1\n{END}\npython3 hard_v1 BM25 only\nno digits here"
    assert find_disallowed_digits(sample) is None
    assert find_disallowed_digits(f"{START}\n1\n{END}\nchunk 220") == "220"


def load_questions():
    raw = QUESTIONS.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    recorded = SHA_FILE.read_text(encoding="utf-8")
    expect = f"{digest}  questions.json\n"
    if recorded != expect:
        sys.exit(
            "hard_v1 hash mismatch: scoring refused\n"
            f"  file {digest}\n  SHA256.txt {recorded!r}"
        )
    doc = json.loads(raw.decode("utf-8"))
    for name, hexdigest in doc["corpus_sha256"].items():
        got = hashlib.sha256((ROOT / "docs" / name).read_bytes()).hexdigest()
        if got != hexdigest:
            sys.exit(f"corpus drift: docs/{name} sha256 {got} != {hexdigest}")
    return doc, digest


def build(max_chars, overlap=OVERLAP):
    docs = rag_demo.parse_docs(rag_demo.DOCS_DIR)
    chunks = []
    for d in docs:
        parts = rag_demo.chunk_text(d["text"], max_chars=max_chars, overlap=overlap)
        for j, text in enumerate(parts):
            chunks.append({
                "chunk_id": f"{d['doc_id']}#c{j}",
                "doc_id": d["doc_id"],
                "title": d["title"],
                "text": text,
                "chars": len(text),
            })
    toks = [rag_demo.bigrams(c["text"]) for c in chunks]
    rag_demo.chunk_tokens_global = toks
    return chunks, rag_demo.TfidfIndex(toks)


def rank_cosine(query, chunks, index):
    qvec = index._vec(rag_demo.bigrams(query))
    if not any(qvec.values()):
        return []
    pairs = index.cosine(qvec)
    pairs.sort(key=lambda si: si[0], reverse=True)
    return [chunks[i] for _, i in pairs[:3]]


def rank_bm25(query, chunks, index):
    pairs = index.bm25(rag_demo.bigrams(query))
    if not pairs or max(s for s, _ in pairs) == 0:
        return []
    pairs.sort(key=lambda si: si[0], reverse=True)
    return [chunks[i] for _, i in pairs[:3]]


def rank_fusion(query, chunks, index, w):
    saved = (rag_demo.W_COS, rag_demo.W_BM25)
    rag_demo.W_COS = w
    rag_demo.W_BM25 = 1.0 - w
    try:
        ranked = rag_demo.retrieve(query, chunks, index, top_k=3)
    finally:
        rag_demo.W_COS, rag_demo.W_BM25 = saved
    return [c for _, c in ranked]


def assert_matches_demo_retrieve():
    """Wiring check on the original five smoke items, not on hard_v1."""
    chunks, index = build(PRIMARY_CHUNK, OVERLAP)
    demo_chunks, demo_index = rag_demo.run_pipeline(verbose=False)
    assert [c["text"] for c in chunks] == [c["text"] for c in demo_chunks]
    rag_demo.W_COS, rag_demo.W_BM25 = 0.6, 0.4
    rag_demo.chunk_tokens_global = [rag_demo.bigrams(c["text"]) for c in demo_chunks]
    for q, _expect in rag_demo.EVAL_SET:
        via_helper = [c["doc_id"] for c in rank_fusion(q, demo_chunks, demo_index, PRIMARY_W)]
        via_demo = [c["doc_id"] for _, c in rag_demo.retrieve(q, demo_chunks, demo_index, top_k=3)]
        assert via_helper == via_demo, (q, via_helper, via_demo)


def score_item(ranked, item):
    ids = [c["doc_id"] for c in ranked]
    if not item["answerable"]:
        return {"abstain": len(ids) == 0, "hit": None, "rank": 0, "docs": ids}
    expect = list(item["expect_docs"])
    if len(expect) == 1:
        want = expect[0]
        for i, doc_id in enumerate(ids, 1):
            if doc_id == want:
                return {"abstain": False, "hit": True, "rank": i, "docs": ids}
        return {"abstain": len(ids) == 0, "hit": False, "rank": 0, "docs": ids}
    need = set(expect)
    seen = set()
    for i, doc_id in enumerate(ids, 1):
        if doc_id in need:
            seen.add(doc_id)
        if seen == need:
            return {"abstain": False, "hit": True, "rank": i, "docs": ids}
    return {"abstain": len(ids) == 0, "hit": False, "rank": 0, "docs": ids}


def rr_of(row):
    if row["hit"]:
        return 1.0 / row["rank"]
    return 0.0


def bootstrap_cell(rows):
    """rows: list of (hit, rank). One generator, shared resample for both metrics."""
    n = len(rows)
    rng = random.Random(SEED)
    recall_reps = []
    mrr_reps = []
    for _ in range(B):
        idxs = [rng.randrange(n) for _ in range(n)]
        hits = 0
        mrr = 0.0
        for i in idxs:
            hit, rank = rows[i]
            if hit:
                hits += 1
                mrr += 1.0 / rank
        recall_reps.append(hits / n)
        mrr_reps.append(mrr / n)
    recall_reps.sort()
    mrr_reps.sort()
    return (
        (recall_reps[K_LO], recall_reps[K_HI]),
        (mrr_reps[K_LO], mrr_reps[K_HI]),
    )


def bootstrap_diff(left, right, metric):
    n = len(left)
    rng = random.Random(SEED)
    reps = []
    for _ in range(B):
        idxs = [rng.randrange(n) for _ in range(n)]
        if metric == "recall":
            a = sum(1.0 for i in idxs if left[i][0]) / n
            b = sum(1.0 for i in idxs if right[i][0]) / n
        else:
            a = sum((1.0 / left[i][1]) if left[i][0] else 0.0 for i in idxs) / n
            b = sum((1.0 / right[i][1]) if right[i][0] else 0.0 for i in idxs) / n
        reps.append(a - b)
    reps.sort()
    return (reps[K_LO], reps[K_HI])


def fmt(x):
    return f"{x:.4f}"


def fmt_ci(pair):
    return f"[{fmt(pair[0])}, {fmt(pair[1])}]"


def cell_from(rows):
    n = len(rows)
    hits = sum(1 for hit, _rank in rows if hit)
    mrr = sum((1.0 / rank) if hit else 0.0 for hit, rank in rows) / n
    recall_ci, mrr_ci = bootstrap_cell(rows)
    return {
        "hits": hits,
        "n": n,
        "recall": fmt(hits / n),
        "recall_ci": fmt_ci(recall_ci),
        "mrr": fmt(mrr),
        "mrr_ci": fmt_ci(mrr_ci),
        "_rows": rows,
        "_mrr": mrr,
        "_recall_ci": recall_ci,
        "_mrr_ci": mrr_ci,
    }


def public_cell(cell):
    return {k: v for k, v in cell.items() if not k.startswith("_")}


def strip_private(obj):
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def rows_for(per_item, items_by_id, chunk, method, category):
    rows = []
    for item in per_item:
        if item["chunk"] != chunk or item["method"] != method:
            continue
        src = items_by_id[item["id"]]
        if not src["answerable"]:
            continue
        if category is not None and src["category"] != category:
            continue
        rows.append((bool(item["hit"]), int(item["rank"])))
    return rows


def smoke_numbers():
    rag_demo.W_COS, rag_demo.W_BM25 = 0.6, 0.4
    docs = rag_demo.parse_docs(rag_demo.DOCS_DIR)
    chunks, index = rag_demo.run_pipeline(verbose=False)
    # evaluate prints; keep the check output to the protocol block.
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        hits, mrr = rag_demo.evaluate(chunks, index)
    return {
        "docs": len(docs),
        "parsed_chars": sum(d["chars"] for d in docs),
        "chunks": len(chunks),
        "vocab": index.vocab_size,
        "hits": hits,
        "n": len(rag_demo.EVAL_SET),
        "mrr": fmt(mrr),
    }


def relation(point_diff, higher):
    if point_diff == 0.0:
        return "tie"
    return "higher" if higher else "lower"


def judge(point_diff, ci, higher):
    win = bool(higher and ci[0] > 0.0)
    return {
        "diff": fmt(point_diff),
        "ci": fmt_ci(ci),
        "relation": relation(point_diff, higher),
        "win": win,
    }


def decision_prose(pairs):
    """Chinese and English from the same flags. pairs keyed by (metric, other)."""
    def both(metric):
        return pairs[(metric, "cosine")]["win"] and pairs[(metric, "bm25")]["win"]

    def line(metric, other, lang):
        row = pairs[(metric, other)]
        if lang == "en":
            if row["win"]:
                return (
                    f"fusion(0.6) beats {other} on {metric}: "
                    f"diff {row['diff']}, paired CI {row['ci']} entirely above 0."
                )
            if row["relation"] == "higher":
                return (
                    f"fusion(0.6) is higher than {other} on {metric} "
                    f"(diff {row['diff']}, paired CI {row['ci']}) but that is not a win."
                )
            if row["relation"] == "tie":
                return f"fusion(0.6) ties {other} on {metric} (diff {row['diff']}, CI {row['ci']})."
            return (
                f"fusion(0.6) is lower than {other} on {metric} "
                f"(diff {row['diff']}, paired CI {row['ci']})."
            )
        if row["win"]:
            return (
                f"fusion(0.6) 在 {metric} 上胜过 {other}："
                f"差 {row['diff']}，配对区间 {row['ci']} 整个在 0 之上。"
            )
        if row["relation"] == "higher":
            return (
                f"fusion(0.6) 在 {metric} 上的点估计高于 {other}"
                f"（差 {row['diff']}，配对区间 {row['ci']}），但按规则不算赢。"
            )
        if row["relation"] == "tie":
            return f"fusion(0.6) 与 {other} 在 {metric} 上打平（差 {row['diff']}，区间 {row['ci']}）。"
        return (
            f"fusion(0.6) 在 {metric} 上低于 {other}"
            f"（差 {row['diff']}，配对区间 {row['ci']}）。"
        )

    en = []
    zh = []
    for metric, label_en, label_zh in (
        ("recall", "recall", "recall"),
        ("mrr", "MRR", "MRR"),
    ):
        if both(metric):
            en.append(f"On {label_en}, fusion(0.6) beats both single methods under the pre-registered rule.")
            zh.append(f"按预注册规则，fusion(0.6) 的 {label_zh} 同时胜过 cosine 和 bm25。")
        else:
            en.append(f"On {label_en}, fusion(0.6) does not beat both single methods.")
            zh.append(f"按预注册规则，fusion(0.6) 的 {label_zh} 没有同时胜过 cosine 和 bm25。")
        for other in ("cosine", "bm25"):
            en.append(line(metric, other, "en"))
            zh.append(line(metric, other, "zh"))
    return zh, en


def render_table(headers, rows):
    head = "| " + " | ".join(headers) + " |"
    bar = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([head, bar, *body])


def render_block(payload):
    smoke = payload["smoke"]
    lines = []
    lines.append(f"hard_v1 questions.json sha256: {payload['question_sha256']}")
    lines.append("corpus sha256 (frozen in the question file, rechecked):")
    for name in sorted(payload["corpus_sha256"]):
        lines.append(f"- {name} {payload['corpus_sha256'][name]}")
    lines.append(
        "Smoke (original five-question quiz, demo defaults, not part of the 28): "
        f"docs {smoke['docs']}, parsed chars {smoke['parsed_chars']}, "
        f"chunks {smoke['chunks']}, vocab {smoke['vocab']}, "
        f"recall@3 {smoke['hits']}/{smoke['n']}, MRR {smoke['mrr']}."
    )
    lines.append(
        "hard_v1 is author-written and small: "
        f"n={payload['n']} "
        f"(paraphrase {payload['category_counts']['paraphrase']}, "
        f"exact_entity {payload['category_counts']['exact_entity']}, "
        f"multi_doc {payload['category_counts']['multi_doc']}, "
        f"unanswerable {payload['category_counts']['unanswerable']}). "
        "Labels were hashed before this scoring run."
    )
    lines.append("Decision (chunk 220, overlap 40, answerable items only):")
    lines.append("中文：")
    lines.extend(f"- {row}" for row in payload["decision_zh"])
    lines.append("English:")
    lines.extend(f"- {row}" for row in payload["decision_en"])
    lines.append("")
    lines.append("Primary methods at chunk 220, all answerable items:")
    lines.append(render_table(
        ["method", "hits", "n", "recall@3", "recall 95% CI", "MRR", "MRR 95% CI"],
        [
            [m, c["hits"], c["n"], c["recall"], c["recall_ci"], c["mrr"], c["mrr_ci"]]
            for m, c in payload["primary_overall"]
        ],
    ))
    lines.append("")
    lines.append("Paired differences at chunk 220, all answerable items (A minus B):")
    lines.append(render_table(
        ["metric", "A", "B", "diff", "paired 95% CI", "relation", "win"],
        [
            [r["metric"], r["a"], r["b"], r["diff"], r["ci"], r["relation"], str(r["win"]).lower()]
            for r in payload["paired"]
        ],
    ))
    lines.append("")
    lines.append("Per category at chunk 220:")
    lines.append(render_table(
        ["category", "method", "hits", "n", "recall@3", "recall 95% CI", "MRR", "MRR 95% CI"],
        [
            [r["category"], r["method"], r["hits"], r["n"], r["recall"], r["recall_ci"], r["mrr"], r["mrr_ci"]]
            for r in payload["by_category_220"]
        ],
    ))
    lines.append("")
    lines.append(
        "Fusion weight sweep at chunk 220. "
        f"w_bm25 is IEEE 1-w_cos (1-0.8 = {1.0 - 0.8!r}, which is not the literal 0.2). "
        "w=0 reranks the cosine top-10; it is not full-corpus BM25."
    )
    lines.append(render_table(
        ["w_cos", "w_bm25", "category", "hits", "n", "recall@3", "recall 95% CI", "MRR", "MRR 95% CI"],
        [
            [r["w_cos"], r["w_bm25"], r["category"], r["hits"], r["n"], r["recall"], r["recall_ci"], r["mrr"], r["mrr_ci"]]
            for r in payload["weight_sweep"]
        ],
    ))
    lines.append("")
    lines.append("Chunk length sweep, overlap 40, three methods:")
    lines.append(render_table(
        ["chunk", "category", "method", "hits", "n", "recall@3", "recall 95% CI", "MRR", "MRR 95% CI"],
        [
            [r["chunk"], r["category"], r["method"], r["hits"], r["n"], r["recall"], r["recall_ci"], r["mrr"], r["mrr_ci"]]
            for r in payload["chunk_sweep"]
        ],
    ))
    lines.append("")
    lines.append(
        "Abstain at chunk 220. A row abstains only when that procedure returns no chunk. "
        "rag_demo.retrieve has no score threshold."
    )
    lines.append(render_table(
        ["method", "answerable abstain", "unanswerable abstain"],
        [
            [r["method"], f"{r['answerable_abstain']}/{r['answerable_n']}", f"{r['unanswerable_abstain']}/{r['unanswerable_n']}"]
            for r in payload["abstain"]
        ],
    ))
    lines.append(payload["abstain_prediction"])
    lines.append(
        "Empty means the query bigram vector had no positive weight. "
        "It is not a score cutoff. Per unanswerable item at chunk 220:"
    )
    lines.append(render_table(
        ["id", "method", "abstain", "returned docs"],
        [
            [r["id"], r["method"], str(r["abstain"]).lower(), ",".join(r["docs"]) if r["docs"] else "(empty)"]
            for r in payload["unanswerable_220"]
        ],
    ))
    lines.append("")
    lines.append(
        "Definition, not a fitted result: a two-doc hit's rank is the covering prefix, "
        "so that rank is at least 2 and multi_doc MRR is at most 0.5000. "
        "0.5000 on that category means every item completed at rank 2."
    )
    drops = [
        r for r in payload["chunk_sweep"]
        if r["category"] == "all" and r["hits"] != r["n"]
    ]
    if drops:
        bits = ", ".join(f"chunk {r['chunk']} {r['method']} {r['hits']}/{r['n']}" for r in drops)
        lines.append(
            "Observational only, not the win rule: all-answerable recall is below the slice size at "
            + bits
            + ". The win rule stays chunk 220 and is not moved onto these cells."
        )
    else:
        lines.append(
            "Observational: every all-answerable cell in the chunk sweep is a full hit. "
            "The win rule stays chunk 220."
        )
    lines.append("")
    if payload["misses"]:
        lines.append("Answerable misses at chunk 220 for cosine, bm25, and fusion(0.6):")
        for row in payload["misses"]:
            lines.append(
                f"- {row['id']} {row['category']} {row['method']} "
                f"got {','.join(row['docs']) if row['docs'] else '(empty)'} "
                f"expect {','.join(row['expect'])}"
            )
    else:
        lines.append("Answerable misses at chunk 220 for cosine, bm25, and fusion(0.6): none.")
    lines.append("")
    lines.append(
        "Answerable rows worse than the best rank this metric allows "
        "(1 for one doc, 2 for two docs), three methods, all three chunk lengths:"
    )
    if payload["shortfalls"]:
        for row in payload["shortfalls"]:
            got = ",".join(row["docs"]) if row["docs"] else "(empty)"
            lines.append(
                f"- chunk {row['chunk']} {row['id']} {row['category']} {row['method']} "
                f"rank {row['rank']} (best {row['best_possible_rank']}) "
                f"got {got} expect {','.join(row['expect'])}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append(
        "Tuning disclosure: 0.6/0.4 and chunk 220 / overlap 40 were already the demo defaults "
        "in rag_demo.py before these 28 questions existed. "
        f"Neighbors {CHUNKS[0]} and {CHUNKS[2]} and weights {WEIGHTS} were written into PROTOCOL.md before scoring. "
        "No weight or chunk length is promoted to a new default from this grid. "
        "Reporting the grid on the same frozen file is a tuning-on-test risk; the win rule is the only comparison that counts."
    )
    lines.append(
        f"Bootstrap: B={B}, seed={SEED}, fresh random.Random per interval, "
        f"k_lo=(25*(B-1))//1000={K_LO}, k_hi=(975*(B-1)+999)//1000={K_HI}."
    )
    return "\n".join(lines)


def find_disallowed_digits(readme_text):
    start = readme_text.find(START)
    end = readme_text.find(END)
    if start < 0 or end < 0 or end < start:
        return "MISSING_MARKERS"
    outside = readme_text[:start] + readme_text[end + len(END):]
    for token in DIGIT_ALLOW:
        outside = outside.replace(token, "")
    digits = []
    buf = []
    for ch in outside:
        if ch.isdigit():
            buf.append(ch)
        elif buf:
            digits.append("".join(buf))
            buf = []
    if buf:
        digits.append("".join(buf))
    return digits[0] if digits else None


def compute():
    self_check_rules()
    assert_matches_demo_retrieve()
    doc, digest = load_questions()
    items = doc["items"]
    by_id = {item["id"]: item for item in items}
    rankers = {
        "cosine": lambda q, chunks, index: rank_cosine(q, chunks, index),
        "bm25": lambda q, chunks, index: rank_bm25(q, chunks, index),
    }
    for w in WEIGHTS:
        rankers[f"fusion_{w:.1f}"] = (
            lambda q, chunks, index, w=w: rank_fusion(q, chunks, index, w)
        )

    per_item = []
    # Chunk sweep uses the three primary methods. Weight sweep is fusion at 220
    # and reuses fusion_0.6 from the chunk pass.
    for max_chars in CHUNKS:
        chunks, index = build(max_chars, OVERLAP)
        methods = ["cosine", "bm25", "fusion_0.6"]
        if max_chars == PRIMARY_CHUNK:
            methods = ["cosine", "bm25"] + [f"fusion_{w:.1f}" for w in WEIGHTS]
        for method in methods:
            fn = rankers[method]
            # BM25 reads the global token list. Rebuild assignment in case a
            # previous retrieve path replaced it.
            rag_demo.chunk_tokens_global = [rag_demo.bigrams(c["text"]) for c in chunks]
            for item in items:
                ranked = fn(item["q"], chunks, index)
                scored = score_item(ranked, item)
                per_item.append({
                    "id": item["id"],
                    "chunk": max_chars,
                    "method": method,
                    "hit": scored["hit"],
                    "rank": scored["rank"],
                    "abstain": scored["abstain"],
                    "docs": scored["docs"],
                })

    categories = ["paraphrase", "exact_entity", "multi_doc"]

    def pack(chunk, method, category):
        cell = cell_from(rows_for(per_item, by_id, chunk, method, category))
        out = public_cell(cell)
        out["_rows"] = cell["_rows"]
        out["_mrr"] = cell["_mrr"]
        return out

    primary_methods = ["cosine", "bm25", "fusion_0.6"]
    primary_overall = []
    primary_rows = {}
    for method in primary_methods:
        cell = pack(PRIMARY_CHUNK, method, None)
        primary_rows[method] = cell
        shown = public_cell(cell)
        primary_overall.append((method, shown))

    paired = []
    pair_flags = {}
    comparisons = [("fusion_0.6", "cosine"), ("fusion_0.6", "bm25"), ("bm25", "cosine")]
    for left, right in comparisons:
        lrows = primary_rows[left]["_rows"]
        rrows = primary_rows[right]["_rows"]
        for metric in ("recall", "mrr"):
            if metric == "recall":
                diff = (sum(1 for h, _ in lrows if h) - sum(1 for h, _ in rrows if h)) / len(lrows)
                higher = sum(1 for h, _ in lrows if h) > sum(1 for h, _ in rrows if h)
            else:
                diff = primary_rows[left]["_mrr"] - primary_rows[right]["_mrr"]
                higher = primary_rows[left]["_mrr"] > primary_rows[right]["_mrr"]
            ci = bootstrap_diff(lrows, rrows, metric)
            judged = judge(diff, ci, higher)
            paired.append({
                "metric": metric,
                "a": left,
                "b": right,
                **judged,
            })
            if left == "fusion_0.6":
                pair_flags[(metric, right)] = judged

    decision_zh, decision_en = decision_prose(pair_flags)

    by_category_220 = []
    for category in categories:
        for method in primary_methods:
            cell = public_cell(pack(PRIMARY_CHUNK, method, category))
            by_category_220.append({"category": category, "method": method, **cell})

    weight_sweep = []
    for w in WEIGHTS:
        method = f"fusion_{w:.1f}"
        for category in [None, *categories]:
            cell = public_cell(pack(PRIMARY_CHUNK, method, category))
            weight_sweep.append({
                "w_cos": f"{w:.1f}",
                "w_bm25": fmt(1.0 - w),
                "category": "all" if category is None else category,
                **cell,
            })

    chunk_sweep = []
    for max_chars in CHUNKS:
        for category in [None, *categories]:
            for method in primary_methods:
                cell = public_cell(pack(max_chars, method, category))
                chunk_sweep.append({
                    "chunk": max_chars,
                    "category": "all" if category is None else category,
                    "method": method,
                    **cell,
                })

    answerable_ids = {item["id"] for item in items if item["answerable"]}
    unanswerable_ids = {item["id"] for item in items if not item["answerable"]}
    abstain = []
    for method in primary_methods:
        rows = [r for r in per_item if r["chunk"] == PRIMARY_CHUNK and r["method"] == method]
        abstain.append({
            "method": method,
            "answerable_abstain": sum(1 for r in rows if r["id"] in answerable_ids and r["abstain"]),
            "answerable_n": len(answerable_ids),
            "unanswerable_abstain": sum(1 for r in rows if r["id"] in unanswerable_ids and r["abstain"]),
            "unanswerable_n": len(unanswerable_ids),
        })
    fusion_unans = next(r for r in abstain if r["method"] == "fusion_0.6")["unanswerable_abstain"]
    if fusion_unans == 0:
        abstain_prediction = (
            "Pre-registered abstain prediction held: fusion(0.6) abstained on 0 unanswerable items. "
            "No score threshold exists, and none was added."
        )
    else:
        abstain_prediction = (
            f"Pre-registered abstain prediction failed: fusion(0.6) abstained on {fusion_unans} "
            f"of {len(unanswerable_ids)} unanswerable items. No score threshold was added after the fact."
        )

    misses = []
    shortfalls = []
    for row in per_item:
        if row["method"] not in primary_methods:
            continue
        src = by_id[row["id"]]
        if not src["answerable"]:
            continue
        best = 1 if len(src["expect_docs"]) == 1 else 2
        worse = (not row["hit"]) or row["rank"] > best
        if row["chunk"] == PRIMARY_CHUNK and not row["hit"]:
            misses.append({
                "id": row["id"],
                "category": src["category"],
                "method": row["method"],
                "docs": row["docs"],
                "expect": src["expect_docs"],
            })
        if worse:
            shortfalls.append({
                "chunk": row["chunk"],
                "id": row["id"],
                "category": src["category"],
                "method": row["method"],
                "rank": row["rank"],
                "best_possible_rank": best,
                "docs": row["docs"],
                "expect": src["expect_docs"],
            })

    unanswerable_220 = []
    for src in items:
        if src["answerable"]:
            continue
        for method in primary_methods:
            row = next(
                r for r in per_item
                if r["id"] == src["id"] and r["chunk"] == PRIMARY_CHUNK and r["method"] == method
            )
            unanswerable_220.append({
                "id": src["id"],
                "method": method,
                "abstain": row["abstain"],
                "docs": row["docs"],
                "q": src["q"],
            })

    # Drop bulky per-item docs from the public audit except misses, but keep
    # ranks so a reader can recompute hits without rerunning. Ranks are ints.
    audit = []
    for row in per_item:
        if row["method"] not in primary_methods and row["chunk"] != PRIMARY_CHUNK:
            continue
        if row["chunk"] != PRIMARY_CHUNK and row["method"] not in primary_methods:
            continue
        audit.append({
            "id": row["id"],
            "chunk": row["chunk"],
            "method": row["method"],
            "hit": row["hit"],
            "rank": row["rank"],
            "abstain": row["abstain"],
            "docs": row["docs"],
        })

    payload = {
        "question_sha256": digest,
        "corpus_sha256": doc["corpus_sha256"],
        "n": doc["n"],
        "category_counts": doc["category_counts"],
        "smoke": smoke_numbers(),
        "decision_zh": decision_zh,
        "decision_en": decision_en,
        "primary_overall": [[m, c] for m, c in primary_overall],
        "paired": paired,
        "by_category_220": by_category_220,
        "weight_sweep": weight_sweep,
        "chunk_sweep": chunk_sweep,
        "abstain": abstain,
        "abstain_prediction": abstain_prediction,
        "unanswerable_220": unanswerable_220,
        "misses": misses,
        "shortfalls": shortfalls,
        "audit": audit,
    }
    payload["readme_block"] = render_block(payload)
    return strip_private(payload)


def dump(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def check_readme(payload):
    if not README.is_file():
        sys.exit("README.md is missing")
    text = README.read_text(encoding="utf-8")
    block = payload["readme_block"]
    needle = f"{START}\n{block}\n{END}"
    if needle not in text:
        sys.exit("README number block does not match a fresh run")
    bad = find_disallowed_digits(text)
    if bad == "MISSING_MARKERS":
        sys.exit("README is missing number markers")
    if bad:
        sys.exit(f"README has a number outside the checked block: {bad}")


def main():
    ap = argparse.ArgumentParser(description="Recompute the frozen hard_v1 ablation")
    ap.add_argument("--write", action="store_true", help="write results.json")
    ap.add_argument("--check", action="store_true", help="fail if results or README drift")
    args = ap.parse_args()
    if args.write and args.check:
        sys.exit("pass only one of --write or --check")
    payload = compute()
    blob = dump(payload)
    if args.write:
        RESULTS.write_text(blob, encoding="utf-8")
        print(payload["readme_block"])
        print(f"\nWROTE {RESULTS.relative_to(ROOT)}")
        return 0
    if not RESULTS.is_file():
        sys.exit("results.json is missing; refuse to treat an unscored tree as checked")
    old = RESULTS.read_text(encoding="utf-8")
    if old != blob:
        import difflib
        diff = "".join(difflib.unified_diff(
            old.splitlines(keepends=True),
            blob.splitlines(keepends=True),
            fromfile="results.json",
            tofile="fresh",
        ))
        sys.stderr.write(diff[:8000])
        sys.exit("results.json drifted from a fresh run")
    check_readme(payload)
    print("hard_v1 check ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
