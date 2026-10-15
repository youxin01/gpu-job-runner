# GPU Job Runner

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB)
![Bash 4+](https://img.shields.io/badge/bash-4%2B-4EAA25)
![Linux](https://img.shields.io/badge/platform-Linux-lightgrey)

[English](README.md) | [简体中文](README.zh-CN.md)

**面向共享 GPU 工作站的轻量级 YAML 任务调度工具。**

把命令写进 YAML，工具会等待空闲 NVIDIA GPU、分配设备、并发执行任务，
在失败后自动重试，并保存每次执行的输出。CPU 任务和远程 API 调用也能使用同一队列，设置 `gpus: 0` 即可。

[快速开始](#快速开始) · [使用方法](#使用方法) · [配置说明](#配置说明) · [查看日志](#查看日志)

## 主要功能

- **GPU 分配**：每个任务申请一张或多张卡，支持指定可用设备和排除设备。
- **并发执行**：有空闲卡就启动后续任务，也可以限制同时运行的任务数。
- **CPU / API 任务**：设置 `gpus: 0`，在没有可见 CUDA 设备的环境中执行命令。
- **失败重试**：失败任务移到队尾重新执行，可以设置重试次数上限。
- **日志与汇总**：每次尝试单独记录输出，同时保存退出码、设备分配和耗时。
- **任务清理**：按 `Ctrl+C`、发送 `SIGTERM` 或关闭终端/tmux 时停止活动任务树，包括嵌套 session 中的后代进程。

## 快速开始

需要 **Linux、Bash 4+ 和 Python 3.9+**。GPU 任务还需要 NVIDIA 驱动及可正常运行的
`nvidia-smi`。训练、推理等任务本身的依赖，需要安装在执行对应命令的环境中。

```bash
git clone https://github.com/youxin01/gpu-job-runner.git
cd gpu-job-runner
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

cp gpu_job_runner.example.yaml jobs.yaml
./gpu_job_runner.sh jobs.yaml
```

示例会检查 GPU 分配并打印设备信息。编辑 `jobs.yaml`，把示例命令换成自己的任务即可。

## 使用方法

1. **编写任务**：为每个任务填写 GPU 数量和 Shell 命令。
2. **启动队列**：运行 `./gpu_job_runner.sh jobs.yaml`。
3. **查看结果**：打开终端打印的日志目录，查看 `summary.tsv`。

例如，在项目目录中执行两个独立的 Python 任务：

```yaml
max_parallel_tasks: 2
max_retries: 2

tasks:
  - name: train
    gpus: 2
    workdir: /absolute/path/to/project
    cmd: |
      torchrun --nproc_per_node=2 train.py

  - name: evaluate
    gpus: 1
    workdir: /absolute/path/to/project
    cmd: |
      python evaluate.py
```

运行前替换路径和命令。这些任务相互独立；GPU 足够时，`evaluate` 可以在 `train` 结束前启动。

CPU 任务或远程 API 调用可以这样配置：

```yaml
max_parallel_tasks: 4
max_retries: 2

tasks:
  - name: cpu_check
    gpus: 0
    cmd: |
      python3 -c 'import os; print("visible GPUs:", repr(os.environ["CUDA_VISIBLE_DEVICES"]))'
```

零 GPU 任务的 `CUDA_VISIBLE_DEVICES` 为空字符串。建议设置有限的 `max_parallel_tasks`，
控制 CPU 任务或 API 请求的并发数量。

### 命令行选项

```bash
# 指定日志目录；每次运行使用单独的目录。
./gpu_job_runner.sh jobs.yaml --log-dir /absolute/path/to/logs/run_01

# 按顺序执行多个配置；某个 runner 返回非零时停止。
./gpu_job_runner.sh first.yaml second.yaml

# 使用自定义 GPU 检测脚本，脚本中需定义 get_free_gpus 函数。
./gpu_job_runner.sh jobs.yaml --gpu-helper /absolute/path/to/get_free_gpus.sh
```

## 配置说明

### 全局字段

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `log_dir` | `./gpu_job_logs/<时间戳>` | 日志和汇总目录；命令行 `--log-dir` 优先。 |
| `max_parallel_tasks` | `0` | 同时运行的任务数上限，包含零 GPU 任务；`0` 表示不限。 |
| `max_retries` | `-1` | 首次执行失败后的重试次数；`-1` 表示无限重试，`0` 表示只执行一次。 |
| `never_use_gpus` | `[]` | 本次运行始终排除的 GPU ID。 |
| `get_free_gpus_extra` | `[]` | GPU 检测工具的参数，每个参数单独作为一个列表项。 |

任务设置了 `workdir` 时，请为 `log_dir` 或 `--log-dir` 使用绝对路径。
当前任务脚本会从任务工作目录解析相对日志路径，可能导致输出未写入日志；未指定时，默认日志目录在内部使用绝对路径。

限制可用的物理 GPU ID：

```yaml
never_use_gpus: [3]
get_free_gpus_extra:
  - --only-use
  - "0,1,2,3"
```

需要在等待期间调整排除设备时，可使用 `--not-use-config`：

```yaml
get_free_gpus_extra:
  - --not-use-config
  - /absolute/path/to/excluded_gpus.yaml
```

该文件必须是 YAML 列表，例如 `[0, 5]`。当前 Shell 工具请使用不含空格的排除文件路径。

### 任务字段

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `name` | 否 | 日志中的任务名；默认依次为 `task_000`、`task_001` 等。 |
| `gpus` | 是 | 非负 GPU 数量；`0` 用于 CPU 任务或远程 API 调用。 |
| `cmd` | 是 | Bash 命令、多行命令块，或命令行列表。 |
| `workdir` | 否 | 执行命令的目录；默认使用启动 runner 时的工作目录。 |

命令通过 Bash 执行，并启用 `set -euo pipefail`。任务继承 runner 的 Python 环境和其他环境变量。
需要其他环境时，可在 `cmd` 中激活，或直接使用对应 Python 解释器的路径。

例如分配到物理 GPU 1 和 3 时，runner 设置 `CUDA_VISIBLE_DEVICES=1,3`，
PyTorch 等框架通常将其映射为任务内的设备 0 和 1。

## 调度与重试

显存占用率和计算利用率均低于 10% 时，GPU 才被视为空闲；runner 也会排除已经分配给自身活动任务的卡。
任务按列表顺序尝试提交：当前任务申请不到足够 GPU 时，后续任务也会等待，包括零 GPU 任务。

任务返回非零退出码后会移到队尾。设置 `max_retries: 2` 时，每个任务最多执行 3 次。
某个任务耗尽重试后，其他任务继续执行；全部结束后 runner 返回状态码 `1`。
默认值 `-1` 会无限重试，建议对可能持续失败的任务设置有限次数。

按 `Ctrl+C` 停止运行。runner 向快照中的任务进程组发送 `TERM`，等待 1 秒后发送 `KILL`，
并以状态码 `130` 退出；再次按 `Ctrl+C` 会强制终止。
若进程在停止前已完全转入后台并被重新托管，则不在可追踪任务树的清理保证内。

<details>
<summary>可选轮询设置</summary>

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GPU_JOB_SCHEDULER_SLEEP_MS` | `6000` | 调度轮询间隔，单位毫秒。 |
| `GPU_JOB_WAIT_LOG_EVERY_MS` | `100000` | 重复输出等待消息的间隔，单位毫秒。 |
| `GPU_JOB_SLEEP_AFTER_MS` | `0` | 检测到任务结束后额外等待的时间，便于显存释放，单位毫秒。 |

```bash
GPU_JOB_SCHEDULER_SLEEP_MS=1000 ./gpu_job_runner.sh jobs.yaml
```

</details>

## 查看日志

runner 会在开始和结束时打印输出目录。

| 文件 | 内容 |
| --- | --- |
| `<序号>_<任务名>.attemptNNN.log` | 单个任务某次尝试的输出。 |
| `scheduler.log` | 等待、分配、完成、重试和停止事件。 |
| `summary.tsv` | 任务名、尝试次数、GPU 数量、设备 ID、退出码、日志路径和耗时。 |
| `exit_codes.tsv` | 任务序号、名称、尝试次数和退出码。 |

每次运行请使用新的目录；复用目录会覆盖汇总文件和同名的尝试日志。
本地 `jobs.yaml`、`gpu_job_logs/`、`logs/` 和 `*.log` 文件已被 Git 忽略。

## 单独获取空闲 GPU

```bash
# 等待两张空闲卡。
python3 get_free_gpus.py --num 2

# 只检查一次；空结果表示空闲卡数量不足。
python3 get_free_gpus.py --num 1 --non-blocking

# 在其他 Shell 脚本中复用。
source ./get_free_gpus.sh
gpu_ids="$(get_free_gpus 2 --only-use 0,1,2,3)"
```

GPU 分配仅在同一个 runner 内协调。轮询检测不提供跨用户、跨 runner 的设备预留；
需要严格隔离时，请使用各自独立的设备白名单或集群调度系统。

## 开发

兼容性要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。

```bash
python3 -m unittest discover -s tests -v
bash -n get_free_gpus.sh gpu_job_runner.sh gpu_job_runner_task.sh
python3 -m py_compile get_free_gpus.py gpu_job_runner_parse.py
```

## 许可证

[MIT License](LICENSE)
