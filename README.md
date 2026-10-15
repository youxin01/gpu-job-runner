# GPU Job Runner

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB)
![Bash 4+](https://img.shields.io/badge/bash-4%2B-4EAA25)
![Linux](https://img.shields.io/badge/platform-Linux-lightgrey)

[English](README.md) | [简体中文](README.zh-CN.md)

**A lightweight YAML job runner for a shared GPU workstation.**

Write your commands in a YAML file. The runner waits for idle NVIDIA GPUs,
assigns devices, runs jobs concurrently, retries failures, and saves each
attempt's output. CPU and remote API tasks can use the same queue with `gpus: 0`.

[Quick start](#quick-start) · [Usage](#usage) · [Configuration](#configuration) · [Logs](#logs)

## Features

- **GPU allocation:** request one or multiple GPUs per task; select allowed devices or exclude specific cards.
- **Concurrent jobs:** launch jobs as GPUs become available and optionally limit the number of active tasks.
- **CPU and API tasks:** use `gpus: 0` to run commands with no visible CUDA devices.
- **Failure retries:** failed jobs move to the end of the queue, with an optional retry limit.
- **Logs and summaries:** keep separate logs for every attempt, plus exit codes, device assignments, and elapsed times.
- **Task cleanup:** stop active task trees on `Ctrl+C`, `SIGTERM`, or terminal/tmux hangup, including descendants in nested sessions.

## Quick start

Requires **Linux, Bash 4+, and Python 3.9+**. GPU jobs also need NVIDIA drivers
and a working `nvidia-smi`. Install your workload's dependencies in the environment
used to run its commands.

```bash
git clone https://github.com/youxin01/gpu-job-runner.git
cd gpu-job-runner
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

cp gpu_job_runner.example.yaml jobs.yaml
./gpu_job_runner.sh jobs.yaml
```

The example checks GPU assignment and prints GPU information. Edit `jobs.yaml`
to replace those commands with your own jobs.

## Usage

1. **Write jobs:** give each task a GPU count and a shell command.
2. **Start the queue:** run `./gpu_job_runner.sh jobs.yaml`.
3. **Inspect results:** open the printed log directory and check `summary.tsv`.

For example, run two independent Python jobs from their project directory:

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

Replace the paths and commands before running. Tasks are independent; `evaluate`
can start while `train` is running if enough GPUs are available.

For CPU work or calls to a remote API:

```yaml
max_parallel_tasks: 4
max_retries: 2

tasks:
  - name: cpu_check
    gpus: 0
    cmd: |
      python3 -c 'import os; print("visible GPUs:", repr(os.environ["CUDA_VISIBLE_DEVICES"]))'
```

A zero-GPU task gets `CUDA_VISIBLE_DEVICES=""`. Set a finite
`max_parallel_tasks` to control concurrent CPU work or API requests.

### Command options

```bash
# Choose a log directory; use a separate directory for each run.
./gpu_job_runner.sh jobs.yaml --log-dir /absolute/path/to/logs/run_01

# Run multiple job files sequentially; stop if one runner exits nonzero.
./gpu_job_runner.sh first.yaml second.yaml

# Use a custom GPU selector shell file defining get_free_gpus.
./gpu_job_runner.sh jobs.yaml --gpu-helper /absolute/path/to/get_free_gpus.sh
```

## Configuration

### Global fields

| Field | Default | Description |
| --- | --- | --- |
| `log_dir` | `./gpu_job_logs/<timestamp>` | Directory for logs and summaries. `--log-dir` takes precedence. |
| `max_parallel_tasks` | `0` | Maximum active tasks, including zero-GPU tasks; `0` means unlimited. |
| `max_retries` | `-1` | Retries after the first attempt; `-1` means unlimited, `0` runs once. |
| `never_use_gpus` | `[]` | GPU IDs excluded for the entire run. |
| `get_free_gpus_extra` | `[]` | GPU selector arguments, with each argument as a separate list item. |

Use an absolute `log_dir` or `--log-dir` when tasks specify `workdir`.
The current wrapper resolves relative log paths from the task's working directory,
which can cause missing task output. The default log directory is absolute internally.

Limit scheduling to selected physical GPU IDs:

```yaml
never_use_gpus: [3]
get_free_gpus_extra:
  - --only-use
  - "0,1,2,3"
```

To update exclusions while the runner is waiting, use `--not-use-config`:

```yaml
get_free_gpus_extra:
  - --not-use-config
  - /absolute/path/to/excluded_gpus.yaml
```

That file must contain a YAML list such as `[0, 5]`. Use a path without spaces
for the current shell helper.

### Task fields

| Field | Required | Description |
| --- | --- | --- |
| `name` | No | Name for logs; defaults to `task_000`, `task_001`, etc. |
| `gpus` | Yes | Non-negative GPU count; `0` is for CPU or remote API work. |
| `cmd` | Yes | Bash command, multiline block, or list of command lines. |
| `workdir` | No | Directory in which to run the command; defaults to the runner's working directory. |

Commands run with Bash and `set -euo pipefail`. Tasks inherit the runner's
Python environment and other environment variables. For a different environment,
activate it inside `cmd` or use its Python executable directly.

If physical GPUs 1 and 3 are assigned, the runner sets
`CUDA_VISIBLE_DEVICES=1,3`; frameworks such as PyTorch normally see local devices 0 and 1.

## Scheduling and retries

A GPU is considered idle when both memory utilization and compute load are below
10%. Devices already assigned to this runner's active tasks are excluded.
Tasks are attempted in list order: if the next task cannot obtain enough GPUs,
later tasks wait too, including zero-GPU tasks.

A nonzero exit code puts the task at the end of the queue. With `max_retries: 2`,
it runs at most three times. Once retries are exhausted, other jobs continue,
and the runner exits with status `1` when all work finishes. The default `-1`
retries indefinitely; set a finite value for jobs that may fail permanently.

Press `Ctrl+C` to stop the run. The runner sends `TERM`, waits one second, then
sends `KILL` to the captured task process groups and exits with status `130`.
A second `Ctrl+C` forces termination. Fully daemonized processes reparented before
shutdown are outside the tracked task-tree cleanup guarantee.

<details>
<summary>Optional polling settings</summary>

| Environment variable | Default | Description |
| --- | --- | --- |
| `GPU_JOB_SCHEDULER_SLEEP_MS` | `6000` | Scheduler polling interval in milliseconds. |
| `GPU_JOB_WAIT_LOG_EVERY_MS` | `100000` | Interval between repeated waiting messages. |
| `GPU_JOB_SLEEP_AFTER_MS` | `0` | Delay after detecting completed jobs to allow GPU memory to settle. |

```bash
GPU_JOB_SCHEDULER_SLEEP_MS=1000 ./gpu_job_runner.sh jobs.yaml
```

</details>

## Logs

The runner prints the output directory when it starts and finishes.

| File | Contents |
| --- | --- |
| `<index>_<name>.attemptNNN.log` | Output from one task attempt. |
| `scheduler.log` | Waiting, assignment, completion, retry, and shutdown events. |
| `summary.tsv` | Task name, attempt, GPU count, device IDs, exit code, log path, and wall time. |
| `exit_codes.tsv` | Task index, name, attempt, and exit code. |

Use a new directory for each run: reusing a directory overwrites summary files
and matching attempt logs. Local `jobs.yaml`, `gpu_job_logs/`, `logs/`, and
`*.log` files are ignored by Git.

## Select idle GPUs directly

```bash
# Wait for two idle GPUs.
python3 get_free_gpus.py --num 2

# Check once; an empty result means insufficient idle GPUs.
python3 get_free_gpus.py --num 1 --non-blocking

# Reuse the selector from a shell script.
source ./get_free_gpus.sh
gpu_ids="$(get_free_gpus 2 --only-use 0,1,2,3)"
```

GPU allocation is coordinated within one runner. Polling does not reserve devices
against other users or runner processes. Use separate device allowlists or a
site scheduler when workloads require strict isolation.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for compatibility requirements.

```bash
python3 -m unittest discover -s tests -v
bash -n get_free_gpus.sh gpu_job_runner.sh gpu_job_runner_task.sh
python3 -m py_compile get_free_gpus.py gpu_job_runner_parse.py
```

## License

[MIT License](LICENSE)
