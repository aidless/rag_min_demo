# B 线（enterprise-rag）迁移准备报告（只设计 + 准备，不建远端仓，不写核心检索代码）

> 冻结声明：`rag_demo.py` 与 `docs/` 为只读冻结，本报告不修改其逻辑与语料。
> 基线数字冻结：recall@3 = 5/5，MRR = 1.000（见 README，任何 B 线改动不得改写该数字，只能新增对照）。

## 1. 脱敏检查（2026-09-10，扫描范围 `rag_min_demo/`）

| 检查项 | 方法 | 结果 |
|---|---|---|
| 真实密钥（sk- / AKIA / Bearer 长 token / 明文 password=） | `grep -rniE` 密钥模式 | **通过：0 命中** |
| `/data1`、Administrator、`C:\Users\Administrator`、`F:\` 绝对路径 | `grep -rn` 路径模式 | **通过：0 命中** |
| Token 变量名残留（TOKEN / API_KEY / SECRET / PASSWORD） | `grep -rn` | **通过：0 命中**（当前 demo 零依赖单机，无需任何凭证） |
| 自指 symlink | `find -type l` | **通过：无 symlink** |
| venv / .venv | 目录检查 | **通过：均不存在** |

结论：**无真实密钥、无绝对路径残留，token 均为无（零变量名），可进入建仓准备。**

## 2. LICENSE 状态

- `rag_min_demo/bline/LICENSE`：MIT（2026，rag_min_demo contributors），已新增。
- 建议新仓根目录放置同一份 MIT LICENSE（本文件即源）。

## 3. 新仓（enterprise-rag）应纳入文件（设计清单，不含核心检索代码）

> 以下为**计划清单**，本次 bounded 任务只落盘 LICENSE + 本报告 + README 小节，
> 不新建核心检索代码文件，避免超出授权。

| 类别 | 计划路径（新仓） | 说明 | 本次是否落盘 |
|---|---|---|---|
| 许可 | `LICENSE` | MIT（由 `bline/LICENSE` 复制） | 源已落盘（bline/LICENSE） |
| 基线复用（只读引用） | `baseline/rag_demo.py` + `baseline/docs/` | 从本仓冻结复制，不改逻辑语料 | 计划，不复制 |
| 配置 | `config/bline.yaml` | B 线参数（权重/TopK/分块上限，设计位） | 计划，不新建代码 |
| 混合检索 | `hybrid/` | 融合策略文档/配置位 | 计划，不写代码 |
| 重排 | `reranker/` | 重排接口文档位 | 计划，不写代码 |
| 评测 harness | `harness/` | 评测驱动文档位 | 计划，不写代码 |
| 延迟 | `latency/` | 延迟记录表位 | 计划，不写代码 |
| golden 集 | `golden/` | 冻结 5 题期望文档（引用基线 EVAL_SET） | 计划，不新建 |
| 评测结果 | `results/` | B 线评测输出位 | 计划，不新建 |
| 演示脚本 | `demo.sh` | 一键演示（调基线 + B 线对照） | 计划，不新建 |
| 本报告 | `BLINE_REPORT.md` | 即本文件 | **已落盘（bline/BLINE_REPORT.md）** |

## 4. 新仓排除项

- `venv/`、`.venv/`、一切虚拟环境与依赖安装产物。
- 自指 symlink（当前无，迁移时禁止新增）。
- 云凭证：一切 `*.pem`、`*credentials*`、console 口令、AK/SK、Bearer token。
- 向量库服务：FAISS/Milvus/Qdrant 服务端与数据卷（B 线只保留接口文档位，不捆绑服务）。
- `/data1` 与 `Administrator` 相关绝对路径（当前 0 残留，CI 建议加 grep 门禁）。

## 5. 是否可建仓迁移

**可以进入建仓迁移准备（本地准备就绪），但不实际建远端仓。**

- 新增文件：`rag_min_demo/bline/LICENSE`、`rag_min_demo/bline/BLINE_REPORT.md`、README B 线小节（见 README 末尾）。
- 未动文件：`rag_demo.py`、`docs/*`（冻结，仅读取）。
- 下一步（需用户另行授权）：`git init enterprise-rag` → 复制 LICENSE/基线只读引用 → 按 §3 逐项建文档位 → 建远端空仓并推送。
