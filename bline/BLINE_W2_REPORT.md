# B线W2报告：golden集扩至32题（SILVER）+ 全网格对照（2026-09-10）

> 冻结声明：`rag_demo.py`（sha256 `f1f26416…a1adab`，与W1报告头一致）与 `docs/`
> 7文档零修改，mtime均为2026-09-09，早于W2全部新增文件（2026-09-10）；
> `rag_min_demo/`非git仓，冻结证据为mtime + sha256 + 基线复跑三件套
> （W2复跑仍为5/5，MRR 1.000）。W2新增仅本目录`w2_*` + 本报告。
> 附带清理：删`rag_min_demo/__pycache__/`（W1遗留字节码，非基线文件，`ls`确认仅删该目录）。

## 1. 做了什么（只新增，不改基线）
| 文件 | 说明 |
|---|---|
| `w2_golden.json` | 32题：5道W1原题（字符串不变保连续）+ 13同义改写 + 14干扰对抗；按文档5/5/5/5/4/4/4（rag_overview/chunking/vector_db/eval/agent_tools/cooking/travel，补上W1零覆盖的agent_tools/travel） |
| `w2_harness.py` | 复用`w1_hybrid`纯函数 + W1网格配置；新增Wilson 95% CI、相对ref（w_cos=0.6）分歧计数、McNemar exact双侧p（stdlib `math.comb`，零新依赖） |
| `w2_results.json` | 全7格点结果（含逐题rank/top1、W1子集分、vs-ref对照、按文档拆分、honesty字段） |

对抗题口径：与干扰文档共享表面词，但答案明确只在期望doc中。真歧义题已排除
（例：“答案忠实度是什么”横跨rag_overview与eval，不入集）。

## 2. 结果（全量报告，无挑选）
7格点**全部相同**：`recall@3=31/32，MRR=0.969，Wilson95=[0.8426,0.9945]`，
W1子集5/5连续性通过。相对ref（w_cos=0.6）：分歧hit=0、分歧rank=0，
McNemar p=1.0（零分歧，无信息量）。按文档（ref策略）：eval 4/5，其余全对。
31个命中**全部为rank 1**，无rank 2/3命中。
- 复算：`PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w2_harness.py`（零依赖，确定性）。

## 3. 唯一miss项标签复核（P3人工核查）
- 题目：“改了链路里的分块模块需要做什么”（paraphrase，expect=eval，实际top1=chunking）。
- 核查：问题问的是“改完模块后做什么”，答案“改动任何一环都要重跑全量评测”只在
  `eval.md`中；`chunking.md`只讲分块策略本身，无此答案。题面含“分块”×2致两路
  检索均被词重叠带偏（cos与BM25原始分已分别核查：两路top均为chunking，见§4）。
- 结论：**标签正确，保留**。这是真实的词重叠陷阱样本，非标注错误。

## 4. 双路原始分抽查（排除harness忽略策略的bug）
抽查3题的cos/BM25原始top-4：分数尺度不同、低位排序偶有差异
（如“拼提示词”题cos第3=agent_tools而BM25第3=eval），但top-1与top-3文档级
结论在32题上完全一致。故“零分歧”是语料真实属性（10 chunk浅池 +
 同源二元组+idf特征），非harness缺陷。语料实测仅10 chunk（非W1估计的约20）。

## 5. 诚实边界（必读）
1. **策略选择仍无区分度**：32题SILVER下纯BM25/纯余弦/RRF文档级结论全同，
   任何“某权重更优”声明仍是无证据强声明，禁止写入README与迁移文档。
2. **唯一信号是表示层共性失败**：miss项两路同错，说明当前字二元组TF-IDF下
   词重叠压倒语义意图——这是双路共享的失败模式，换融合权重治不了。
3. **doc级top-3在10 chunk池上太粗**：top-3即覆盖30% chunk，天花板过高；
   且31/32命中全为rank 1，MRR≈recall， rank信息量为零。
4. **W3方向（三选一，需立项）**：(a) 扩语料文档数与chunk池深度，加真正干扰项；
   (b) 改chunk级评测 + 主看top-1/MRR；(c) 针对“两路top-1分歧”构造对抗题
   （需先扫出两路分歧题再写golden，顺序不可反）。SILVER结论限于本语料，
   不做生产权重推荐（沿用基线0.6/0.4为“沿用值”）。
