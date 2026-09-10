# B线W3报告：扩语料 + chunk级双口径 + 对抗集全网格实测

结论（无信号，诚实判负）：v3-44全集上7格点命中全同（doc 42/44，chunk 42/44），
相对ref（w_cos=0.6）McNemar exact p全为1.0。对抗adv12子集12/12全格点全同。
权重选择在本基线+本语料+本评测口径下无区分度，不做生产权重推荐。

## 实测配置
- 语料：extended 35 chunk（docs 10 + corpus 25，SNAPSHOT按P1勘误排除，w3_p4.py复算断言35）
- 集：w3-v3-44（seed32标签冻结 + adv12），top_k与候选数取w1_config.json
- 网格：w_cos ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} + rrf_k=60，共7格点
- 复算：PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w3_p4.py → w3_results.json
- git不可用（非git仓库），git_sha记unknown；基线mtime仍09-09（W3动基线=FAIL门禁）

## 全集结果（full44）
| 格点 | doc recall | doc MRR | chunk recall | chunk MRR | vs ref hit分歧 | McNemar p |
|---|---|---|---|---|---|---|
| w_cos=0.0 | 42/44 | 0.9280 | 42/44 | 0.9167 | 0 | 1.0 |
| w_cos=0.2 | 42/44 | 0.9280 | 42/44 | 0.9167 | 0 | 1.0 |
| w_cos=0.4 | 42/44 | 0.9280 | 42/44 | 0.9167 | 0 | 1.0 |
| w_cos=0.6（ref） | 42/44 | 0.9280 | 42/44 | 0.9167 | — | — |
| w_cos=0.8 | 42/44 | 0.9280 | 42/44 | 0.9167 | 0 | 1.0 |
| w_cos=1.0 | 42/44 | 0.9167 | 42/44 | 0.9053 | 0 | 1.0（rank分歧各1） |
| rrf_k=60 | 42/44 | 0.9167 | 42/44 | 0.9053 | 0 | 1.0（rank分歧各1） |

Wilson95（ref，doc 42/44）：见w3_results.json（wilson95_doc逐格落盘）。
adv12子集：7格点全部12/12。sens41（剔除3漂移题）：7格点全部40/41。

## 失败归因（2 miss，均为“新语料抢位”，与融合权重无关）
1. seed-Q16《golden集应该包含什么》（expect eval#c0）：extended池下recall_debug#c1（1.0000）置顶，
   P2已报的预期干扰，v3 drift_flags第三条。
2. seed-Q17《改了链路里的分块模块需要做什么》（expect eval#c1）：chunking#c0（1.0000，含“分块太大…”）置顶，
   问法与chunking文档字面重叠过高，属P1扩语料引入的标签-语料失配。
两题在7格点下错得完全一样（hit分歧=0），证明不是权重能修的问题。

## 漂移备注（v3 drift_flags）
- Q14长短文本 / 翻车题在extended池下反而命中（embedding_select#c1、hybrid_retrieval#c0同样作答），
  标签维持原判；sens41剔除后40/41，结论不变（仍全格点全同）。

## 诚实边界
- 自然问法空间56探针穷尽仍零分歧（P3）；本报告只证明“此基线+此语料+此口径下权重无信号”，
  不证明“融合权重一般无用”。
- chunk总数35、seed标签冻结、adv单答案可复核三项可独立复算；W2回放门禁仍在w3_p2_check.json中PASS。
