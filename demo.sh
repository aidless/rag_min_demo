#!/usr/bin/env bash
# 一键演示：全链路评测 + 单查询演示（前提：在本仓根执行）。
set -euo pipefail
cd "$(dirname "$0")"
python3 rag_demo.py
echo "--- 单查询演示 ---"
python3 rag_demo.py --query "分块太大有什么坏处"
