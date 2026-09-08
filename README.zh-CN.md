# GPU Job Runner

一个面向共享 NVIDIA GPU 工作站的轻量级 YAML 任务调度工具。

它会等待空闲 GPU，通过 `CUDA_VISIBLE_DEVICES` 分配设备，并发执行多个任务，
在任务失败后自动重新排队，同时为每次尝试保存独立日志。适用于没有 Slurm 等
集群调度系统、但需要多个实验共享单机 GPU 的场景。

## 主要功能

- 同时根据显存占用率和计算利用率判断 GPU 是否空闲。
- 每个任务可以申请一张或多张 GPU。
- GPU 充足时并发执行，并支持限制最大并发任务数。
- 支持 GPU 白名单、静态排除列表和动态排除文件。
- 失败任务自动移到队尾，持续重试直至成功。
- 收到 `Ctrl+C` 或 `SIGTERM` 时停止 runner 和全部活动子任务。
- 保存每次尝试的日志、调度日志、退出码和耗时汇总。
- 可以依次执行多个 YAML 文件。

## 安装

要求 Linux、NVIDIA 驱动、Bash 4+ 和 Python 3.9+。

```bash
git clone git@github.com:youxin01/gpu-job-runner.git
cd gpu-job-runner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

```bash
cp gpu_job_runner.example.yaml jobs.yaml
./gpu_job_runner.sh jobs.yaml
```

最小配置：

```yaml
log_dir: ./gpu_job_logs/demo
max_parallel_tasks: 2

tasks:
  - name: train_model
    gpus: 2
    workdir: /path/to/project
    cmd: |
      torchrun --nproc_per_node=2 train.py

  - name: evaluate_model
    gpus: 1
    cmd: |
      python evaluate.py
```

任务启动前会自动设置 `CUDA_VISIBLE_DEVICES`。例如任务分配到物理 GPU 1 和 3，
环境变量为 `1,3`，PyTorch 等框架通常会将其映射为任务内的设备 0 和 1。

## 全局字段

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `log_dir` | `./gpu_job_logs/<时间戳>` | 日志和汇总目录。 |
| `max_parallel_tasks` | `0` | 最大并发数；`0` 表示只受 GPU 数量限制。 |
| `never_use_gpus` | `[]` | 本次运行永远不使用的 GPU ID。 |
| `get_free_gpus_extra` | `[]` | 传给 GPU 检测工具的额外参数，每个 token 单独一项。 |

只允许使用指定 GPU：

```yaml
get_free_gpus_extra:
  - --only-use
  - "0,1,2,3"
```

使用动态排除文件：

```yaml
get_free_gpus_extra:
  - --not-use-config
  - /path/to/excluded_gpus.yaml
```

排除文件必须是 GPU ID 的 YAML 列表，例如 `[0, 5]`。

## 任务字段

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `name` | 否 | 用于日志文件的任务名。 |
| `gpus` | 是 | 任务申请的 GPU 数量。 |
| `cmd` | 是 | Shell 命令或多行命令块。 |
| `workdir` | 否 | 执行任务前进入的工作目录。 |

任务使用 Bash 执行，并启用 `set -euo pipefail`。

## 调度与重试

默认情况下，显存占用率和计算利用率均低于 10% 的 GPU 才会被视为空闲。
runner 也会排除已经分配给自身活动任务的 GPU。

任务返回非零退出码时会被放到队列末尾，并持续重试。重试次数没有上限；如果是
配置或代码错误，需要通过 `Ctrl+C` 停止 runner 后修复。

可选环境变量：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GPU_JOB_SLEEP_AFTER_MS` | `0` | 任务结束后等待显存释放的时间。 |
| `GPU_JOB_SCHEDULER_SLEEP_MS` | `6000` | 调度轮询间隔。 |
| `GPU_JOB_WAIT_LOG_EVERY_MS` | `100000` | 重复输出等待状态的间隔。 |

## 停止任务

按一次 `Ctrl+C` 会停止调度器和它启动的全部活动子任务。runner 会先快照活动任务树
中的全部进程组，包括通过 `setsid` 创建了嵌套 session 的后代；随后发送 `TERM`，
等待1秒，再对同一快照发送 `KILL`，回收直接子进程后以状态码130退出。再次按
`Ctrl+C` 会立即强制终止。若进程在中断前已经完全 daemonize 并被重新托管，它已不在
可追踪任务树中，不属于这一保证。

## 单独获取空闲 GPU

```bash
python3 get_free_gpus.py --num 2
python3 get_free_gpus.py --num 1 --non-blocking

source ./get_free_gpus.sh
gpu_ids="$(get_free_gpus 2 --only-use 0,1,2,3)"
```

## 输出文件

- `<序号>_<任务名>.attemptNNN.log`：单次尝试的标准输出和错误输出。
- `scheduler.log`：等待、提交、重试、完成和中断事件。
- `summary.tsv`：GPU 分配、退出码、日志路径和耗时。
- `exit_codes.tsv`：精简的退出码记录。

## 使用边界

该工具可以避免同一个 runner 内部重复分配 GPU，但 GPU 状态检测不是跨用户的
原子锁。其他用户仍可能在检测后抢占同一张卡。需要严格资源隔离时，应配合 GPU
白名单、共享排除文件或 Slurm 等集群调度系统。

## License

[MIT](LICENSE)
