# GPU Job Runner

A lightweight, YAML-driven job runner for shared NVIDIA GPU workstations.

GPU Job Runner waits for idle GPUs, assigns them through `CUDA_VISIBLE_DEVICES`,
runs independent jobs concurrently, retries failed jobs, and records every
attempt. It is intended for a single Linux host where several experiments share
GPUs without a cluster scheduler.

[中文文档](README.zh-CN.md)

## Features

- Detects idle GPUs using memory and compute utilization.
- Supports one or multiple GPUs per task.
- Runs jobs concurrently when resources are available.
- Supports GPU allowlists and exclusion lists.
- Requeues failed jobs at the end of the queue until they succeed.
- Stops the runner and all active child tasks on `Ctrl+C` or `SIGTERM`.
- Writes per-attempt logs, scheduling events, exit codes, and wall times.
- Executes multiple YAML job files sequentially.

## Requirements

- Linux with NVIDIA drivers and `nvidia-smi`
- Bash 4 or newer
- Python 3.9 or newer

Install the Python dependencies:

```bash
git clone git@github.com:youxin01/gpu-job-runner.git
cd gpu-job-runner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
cp gpu_job_runner.example.yaml jobs.yaml
./gpu_job_runner.sh jobs.yaml
```

A minimal configuration:

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

The runner sets `CUDA_VISIBLE_DEVICES` before launching each command. If a task
receives physical GPUs 1 and 3, the command sees `CUDA_VISIBLE_DEVICES=1,3` and
frameworks such as PyTorch normally expose them as local devices 0 and 1.

## YAML Reference

### Global fields

| Field | Default | Description |
| --- | --- | --- |
| `log_dir` | `./gpu_job_logs/<timestamp>` | Log and summary directory. |
| `max_parallel_tasks` | `0` | Concurrent job limit; `0` means no explicit limit. |
| `never_use_gpus` | `[]` | GPU IDs that this run must never select. |
| `get_free_gpus_extra` | `[]` | Arguments forwarded to the GPU selector, one token per item. |

Restrict scheduling to selected GPUs:

```yaml
get_free_gpus_extra:
  - --only-use
  - "0,1,2,3"
```

Or provide a YAML file containing excluded GPU IDs:

```yaml
get_free_gpus_extra:
  - --not-use-config
  - /path/to/excluded_gpus.yaml
```

The exclusion file must contain a list, for example `[0, 5]`.

### Task fields

| Field | Required | Description |
| --- | --- | --- |
| `name` | No | Name used in logs; defaults to the task index. |
| `gpus` | Yes | Number of GPUs requested by the command. |
| `cmd` | Yes | Shell command or multiline shell block. |
| `workdir` | No | Working directory used to run the command. |

Task commands run with Bash and `set -euo pipefail`.

## Scheduling and Retries

A GPU is considered idle when both its memory utilization and compute load are
below 10%. The scheduler also excludes GPUs already assigned to its own active
tasks.

When a command exits with a nonzero status, it is appended to the end of the
queue and retried. Retries are intentionally unlimited, so configuration or code
errors continue retrying until the runner is stopped.

Optional timing controls:

| Environment variable | Default | Description |
| --- | --- | --- |
| `GPU_JOB_SLEEP_AFTER_MS` | `0` | Delay after a job exits, allowing GPU memory to settle. |
| `GPU_JOB_SCHEDULER_SLEEP_MS` | `6000` | Scheduler polling interval. |
| `GPU_JOB_WAIT_LOG_EVERY_MS` | `100000` | Interval between repeated waiting messages. |

## Stopping a Run

Press `Ctrl+C` once to stop the scheduler and every active child task. The runner
snapshots process groups for the active task trees, including descendants that
created nested sessions with `setsid`. It sends `TERM`, waits one second, sends
`KILL` to the same snapshot, reaps its direct children, and exits with status
130. Pressing `Ctrl+C` again forces immediate termination. A process that fully
daemonized and was reparented before the interrupt is outside this task-tree
guarantee.

## Selecting GPUs Directly

The GPU selector can also be used independently:

```bash
# Wait until two GPUs are idle.
python3 get_free_gpus.py --num 2

# Check once; print an empty line if no GPU is available.
python3 get_free_gpus.py --num 1 --non-blocking

# Use the shell function from another script.
source ./get_free_gpus.sh
gpu_ids="$(get_free_gpus 2 --only-use 0,1,2,3)"
```

## Logs

Each run writes:

- `<index>_<name>.attemptNNN.log`: stdout and stderr from one attempt.
- `scheduler.log`: waiting, assignment, retry, completion, and interrupt events.
- `summary.tsv`: GPU assignment, exit status, log path, and wall time.
- `exit_codes.tsv`: compact per-attempt exit statuses.

## Operational Boundary

GPU Job Runner coordinates jobs launched by the same runner process. GPU polling
is not an atomic reservation against unrelated users, so another process may
claim a GPU after detection. Use an allowlist, a shared exclusion file, or a
site-level scheduler when strict isolation is required.

## License

[MIT](LICENSE)
