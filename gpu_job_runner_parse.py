#!/usr/bin/env python3
"""Parse gpu_job_runner YAML config into a staging directory (one subdir per task)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    print("需要 PyYAML: pip install pyyaml", file=sys.stderr)
    raise SystemExit(1) from e


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path, help="YAML 配置文件路径")
    p.add_argument("staging", type=Path, help="输出 staging 目录（会被清空后重建）")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "tasks" not in data:
        print("配置无效: 根节点需为 dict 且包含 tasks 列表", file=sys.stderr)
        raise SystemExit(1)

    tasks = data["tasks"]
    if not isinstance(tasks, list) or not tasks:
        print("tasks 必须为非空列表", file=sys.stderr)
        raise SystemExit(1)

    if args.staging.exists():
        shutil.rmtree(args.staging)
    args.staging.mkdir(parents=True)

    # Global options written for bash reader
    log_dir = data.get("log_dir", "")
    gpu_extra = data.get("get_free_gpus_extra", [])
    never_use_gpus = data.get("never_use_gpus", [])
    max_parallel_tasks = data.get("max_parallel_tasks", 0)
    if gpu_extra is None:
        gpu_extra = []
    if never_use_gpus is None:
        never_use_gpus = []
    if not isinstance(gpu_extra, list):
        print("get_free_gpus_extra 必须是字符串列表", file=sys.stderr)
        raise SystemExit(1)
    if not isinstance(never_use_gpus, list):
        print("never_use_gpus 必须是 GPU ID 列表", file=sys.stderr)
        raise SystemExit(1)
    try:
        never_use_gpus = [int(x) for x in never_use_gpus]
    except (TypeError, ValueError):
        print("never_use_gpus 必须只包含整数 GPU ID", file=sys.stderr)
        raise SystemExit(1)
    try:
        max_parallel_tasks = int(max_parallel_tasks)
    except (TypeError, ValueError):
        print("max_parallel_tasks 必须是整数（0 表示不限制）", file=sys.stderr)
        raise SystemExit(1)
    if max_parallel_tasks < 0:
        print("max_parallel_tasks 不能小于 0（0 表示不限制）", file=sys.stderr)
        raise SystemExit(1)

    (args.staging / "_global.log_dir").write_text(str(log_dir).strip() + "\n", encoding="utf-8")
    (args.staging / "_global.gpu_extra").write_text(
        "\n".join(str(x) for x in gpu_extra) + "\n", encoding="utf-8"
    )
    (args.staging / "_global.never_use_gpus").write_text(
        "\n".join(str(x) for x in never_use_gpus) + "\n", encoding="utf-8"
    )
    (args.staging / "_global.max_parallel_tasks").write_text(
        str(max_parallel_tasks) + "\n", encoding="utf-8"
    )

    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            print(f"tasks[{i}] 必须是 dict", file=sys.stderr)
            raise SystemExit(1)
        raw_name = str(t.get("name", f"task_{i:03d}"))
        for ch in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
            raw_name = raw_name.replace(ch, "_")
        name = raw_name.strip() or f"task_{i:03d}"
        if "gpus" not in t or "cmd" not in t:
            print(f"任务 {name}: 缺少 gpus 或 cmd", file=sys.stderr)
            raise SystemExit(1)
        gpus = int(t["gpus"])
        cmd = t["cmd"]
        if isinstance(cmd, list):
            cmd = "\n".join(str(x) for x in cmd)
        else:
            cmd = str(cmd)
        workdir = t.get("workdir", "") or ""

        td = args.staging / f"{i:03d}_{name}"
        td.mkdir(parents=True)
        (td / "gpus").write_text(str(gpus) + "\n", encoding="utf-8")
        (td / "name").write_text(name + "\n", encoding="utf-8")
        (td / "cmd.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + cmd + "\n", encoding="utf-8")
        (td / "cmd.sh").chmod(0o755)
        (td / "workdir").write_text(workdir + "\n", encoding="utf-8")

    (args.staging / "_task_count").write_text(str(len(tasks)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
