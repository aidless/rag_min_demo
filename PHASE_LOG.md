# P2 PHASE_LOG — B线 rag_min_demo（世界先进封顶）

- 结果 vs 验收 G2：基线 `rag_demo.py` recall@3=5/5 MRR=1.000（重跑一致）；W1 全网格 7 格点 PASS；W2 分歧 0 PASS；W3 longq/p3b（320行零分歧）/p3c（11行零分歧）/termsat 全 PASS。根 `LICENSE`（与 bline 逐字节一致）+ `.gitignore` 已补。G2 通过。
- 产出物：`rag_min_demo/LICENSE`、`rag_min_demo/.gitignore`。
- 失败+纠正：无（一次通过）。B线绝对路径 grep 命中仅为报告文档自述检查方法的字样，非泄露。
- 未动：`rag_demo.py`、`docs/`（冻结）；未建远端仓（需另行授权）。
