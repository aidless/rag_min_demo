.PHONY: check eval

# Offline recompute of every number in the README block. Stdlib only.
check:
	python3 evalset/hard_v1/run_ablation.py --check

# --- acceptance_backfill (EVAL_ACCEPTANCE_STANDARD.md P4) ---
# 证据来源: verify.yml
# 命令逐字取自该来源，未增删任何参数。
# 归一: python → python3（本机无 python 别名，实测 Error 127）
eval:
	python3 -m pytest -q
