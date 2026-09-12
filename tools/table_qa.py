#!/usr/bin/env python3
"""StructGPT 式表问答（B 线）——补上 P3 未迁移项。

原设计稿的 B 线「StructGPT 稀疏 KG → 表问答分支」当时未迁（B 线无表格语料）。
本模块补上完整可跑的两阶段管线 + EM 度量，并配最小确定性 fixture：

  阶段一 schema 选择：按查询词与列名的字面/别名重叠挑出候选列
  阶段二 cell 定位：在候选列上定位行，取出目标单元格；或做列聚合

支持的问题类型（确定性、可 EM 判定）：
  lookup    —— 某行某列的取值（如「alice 的城市」）
  count     —— 满足条件的行数（如「RD 有多少人」）
  sum / avg —— 某列在筛选子集上的求和/均值

诚实边界：这是**小表确定性管线**，不是 LLM 语义解析；不声称对任意自然语言的泛化。
零依赖：标准库 only。
"""

import json
import os
import re

__all__ = ["load_table", "normalize", "em", "answer"]

_HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(_HERE, "table_fixture.json")

# 中文别名（fixture 用英文列名；查询常为中文）
ALIAS = {
    "name": ("name", "名字", "姓名", "谁"),
    "dept": ("dept", "部门", "组"),
    "city": ("city", "城市", "地方"),
    "salary": ("salary", "薪资", "工资", "薪水"),
    "years": ("years", "年限", "年资", "工龄"),
}
AGG = {"sum": ("总和", "合计", "总共", "sum", "total"),
       "avg": ("平均", "均值", "avg", "mean", "average"),
       "count": ("多少", "几个", "人数", "count", "how many")}


def load_table(path=FIXTURE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize(s):
    """EM 归一：小写、去空白/标点、数字规范化（25 与 25.0 等价）。

    注意：**先试数值解析**再退到去标点——否则小数点会被当标点剔除，
    使 "2.0" 变成 "20" 而不等于 "2"（本项曾是真 bug，已修）。
    """
    if s is None:
        return ""
    t = str(s).strip().lower()

    for cand in (re.sub(r"\s+", "", t),
                 re.sub(r"[\s，,。.；;：:！!？?、\"'`]+", "", t)):
        if not cand:
            continue
        try:
            f = float(cand)
            return str(int(f)) if f.is_integer() else str(f)
        except ValueError:
            continue
    return re.sub(r"[\s，,。.；;：:！!？?、\"'`]+", "", t)


def em(pred, gold):
    return normalize(pred) == normalize(gold)


def _find_col(query, columns):
    """阶段一：schema 选择——返回命中的列名（最后一个命中为「目标列」）。"""
    hits = []
    for c in columns:
        for a in ALIAS.get(c, (c,)):
            if a in query:
                hits.append(c)
                break
    return hits


def _find_row(table, query, columns, row_filter_col=None):
    """阶段二辅助：按某列的值匹配找出目标行（用于 lookup）。"""
    idx = {c: i for i, c in enumerate(columns)}
    q = query.lower()
    for row in table["rows"]:
        if row_filter_col:
            i = idx[row_filter_col]
            if str(row[i]).lower() in q:
                return row
        else:
            for i, cell in enumerate(row):
                if str(cell).lower() in q:
                    return row
    return None


def _agg(query):
    for kind, words in AGG.items():
        if any(w in query for w in words):
            return kind
    return None


def answer(table, query):
    """返回 (答案字符串, 解析出的计划 dict)；无法解析则答案为 None。"""
    columns = table["columns"]
    idx = {c: i for i, c in enumerate(columns)}
    cols_hit = _find_col(query, columns)
    kind = _agg(query)
    plan = {"cols": cols_hit, "agg": kind}

    # 过滤子集：若查询里出现某列的某个值，则先筛选
    subset = list(table["rows"])
    for c, i in idx.items():
        for row in table["rows"]:
            v = str(row[i])
            if len(v) >= 2 and v.lower() in query.lower():
                subset = [r for r in subset if str(r[i]).lower() == v.lower()]
                plan["filter"] = {c: v}
                break

    if kind == "count":
        plan["target"] = "rows"
        return str(len(subset)), plan

    if kind in ("sum", "avg"):
        target = None
        for c in reversed(cols_hit):
            if isinstance(table["rows"][0][idx[c]], (int, float)):
                target = c
                break
        if target is None:
            return None, plan
        vals = [r[idx[target]] for r in subset]
        plan["target"] = target
        if not vals:
            return None, plan
        if kind == "sum":
            v = sum(vals)
            return (str(int(v)) if float(v).is_integer() else str(v)), plan
        a = sum(vals) / len(vals)
        return (str(round(a, 4)).rstrip("0").rstrip(".") or "0"), plan

    # lookup：目标列 = 最后一个命中列；行 = 查询里出现的其它列值所在行
    if not cols_hit:
        return None, plan
    target = cols_hit[-1]
    row = _find_row(table, query, columns)
    plan["target"] = target
    if row is None:
        return None, plan
    v = row[idx[target]]
    return (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)), plan
