#!/usr/bin/env bash
# gpu_job_runner.sh — 并发调度多个 GPU 任务，复用 get_free_gpus.sh 占卡/等待逻辑
#
# 用法：
#   ./gpu_job_runner.sh /path/to/jobs.yaml
#   ./gpu_job_runner.sh /path/to/jobs.yaml --log-dir /path/to/logs
#   ./gpu_job_runner.sh /path/to/jobs.yaml --gpu-helper /path/to/get_free_gpus.sh
#
# 环境变量（可选）：
#   GPU_JOB_SLEEP_AFTER_MS=2000     有任务结束后等待若干毫秒再继续调度（便于显存回落）
#   GPU_JOB_SCHEDULER_SLEEP_MS=1000 调度器轮询间隔（毫秒）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_HELPER="${SCRIPT_DIR}/get_free_gpus.sh"
CONFIGS=()
LOG_DIR_OVERRIDE=""

usage() {
    echo "用法: $0 <jobs1.yaml> [jobs2.yaml ...] [--log-dir DIR] [--gpu-helper PATH]" >&2
    echo "" >&2
    echo "配置文件格式见: ${SCRIPT_DIR}/gpu_job_runner.example.yaml" >&2
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --log-dir)
            LOG_DIR_OVERRIDE="${2:?需要 --log-dir 路径}"
            shift 2
            ;;
        --gpu-helper)
            GPU_HELPER="${2:?需要 --gpu-helper 路径}"
            shift 2
            ;;
        -*)
            echo "未知参数: $1" >&2
            usage
            exit 1
            ;;
        *)
            CONFIGS+=("$1")
            shift
            ;;
    esac
done

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "请至少提供一个 jobs.yaml 路径" >&2
    usage
    exit 1
fi
if [ ! -f "$GPU_HELPER" ]; then
    echo "找不到 get_free_gpus.sh: $GPU_HELPER" >&2
    exit 1
fi

for cfg in "${CONFIGS[@]}"; do
    if [ ! -f "$cfg" ]; then
        echo "配置文件不存在: $cfg" >&2
        exit 1
    fi
done

if [ "${#CONFIGS[@]}" -gt 1 ]; then
    self_path="$(readlink -f "$0")"
    for cfg in "${CONFIGS[@]}"; do
        echo "========================================"
        echo "开始顺序执行: $cfg"
        echo "========================================"
        args=("$cfg")
        if [ -n "$LOG_DIR_OVERRIDE" ]; then
            args+=("--log-dir" "$LOG_DIR_OVERRIDE")
        fi
        if [ -n "$GPU_HELPER" ]; then
            args+=("--gpu-helper" "$GPU_HELPER")
        fi
        bash "$self_path" "${args[@]}"
    done
    exit 0
fi

CONFIG="${CONFIGS[0]}"

# shellcheck source=/dev/null
source "$GPU_HELPER"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
STAGING="$(mktemp -d "${TMPDIR:-/tmp}/gpu_job_runner.${RUN_TS}.XXXXXX")"
cleanup() { rm -rf "$STAGING"; }
trap cleanup EXIT

python3 "${SCRIPT_DIR}/gpu_job_runner_parse.py" "$CONFIG" "$STAGING"

GLOBAL_LOG_DIR="$(cat "${STAGING}/_global.log_dir" | tr -d '\r' | head -1 || true)"
if [ -n "$LOG_DIR_OVERRIDE" ]; then
    RUN_LOG_DIR="$LOG_DIR_OVERRIDE"
elif [ -n "$GLOBAL_LOG_DIR" ]; then
    RUN_LOG_DIR="$GLOBAL_LOG_DIR"
else
    RUN_LOG_DIR="${PWD}/gpu_job_logs/${RUN_TS}"
fi
mkdir -p "$RUN_LOG_DIR"

SUMMARY="${RUN_LOG_DIR}/summary.tsv"
RESULTS="${RUN_LOG_DIR}/exit_codes.tsv"
SCHEDULER_LOG="${RUN_LOG_DIR}/scheduler.log"
echo -e "index\tname\tattempt\tgpus\tcuda_visible\texit_code\tlog_file\twall_secs" >"$SUMMARY"
echo -e "index\tname\tattempt\texit_code" >"$RESULTS"
: >"$SCHEDULER_LOG"

GPU_EXTRA_ARGS=()
while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    GPU_EXTRA_ARGS+=("$line")
done < "${STAGING}/_global.gpu_extra"

STATIC_NEVER_USE=()
while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    STATIC_NEVER_USE+=("$line")
done < "${STAGING}/_global.never_use_gpus"

sleep_after_ms="${GPU_JOB_SLEEP_AFTER_MS:-0}"
scheduler_sleep_ms="${GPU_JOB_SCHEDULER_SLEEP_MS:-6000}"
wait_log_every_ms="${GPU_JOB_WAIT_LOG_EVERY_MS:-100000}"
max_parallel_tasks="$(tr -d '\r\n' < "${STAGING}/_global.max_parallel_tasks" 2>/dev/null || true)"
if [ -z "$max_parallel_tasks" ]; then
    max_parallel_tasks=0
fi
if ! [[ "$max_parallel_tasks" =~ ^[0-9]+$ ]]; then
    echo "配置错误: max_parallel_tasks 必须是非负整数（0 表示不限制）" >&2
    exit 1
fi

BASE_ONLY_USE=""
BASE_NOT_USE_CONFIG=""
GPU_PASSTHROUGH_ARGS=()
i=0
while [ "$i" -lt "${#GPU_EXTRA_ARGS[@]}" ]; do
    arg="${GPU_EXTRA_ARGS[$i]}"
    case "$arg" in
        --nowait|--non-blocking)
            ;;
        --only-use)
            i=$((i + 1))
            if [ "$i" -ge "${#GPU_EXTRA_ARGS[@]}" ]; then
                echo "get_free_gpus_extra 中的 --only-use 缺少值" >&2
                exit 1
            fi
            BASE_ONLY_USE="${GPU_EXTRA_ARGS[$i]}"
            ;;
        --not-use-config)
            i=$((i + 1))
            if [ "$i" -ge "${#GPU_EXTRA_ARGS[@]}" ]; then
                echo "get_free_gpus_extra 中的 --not-use-config 缺少值" >&2
                exit 1
            fi
            BASE_NOT_USE_CONFIG="${GPU_EXTRA_ARGS[$i]}"
            ;;
        *)
            GPU_PASSTHROUGH_ARGS+=("$arg")
            ;;
    esac
    i=$((i + 1))
done

RUNTIME_NOT_USE_CONFIG="${STAGING}/_runtime.not_use.yaml"

declare -A RESERVED_GPU_SET=()
declare -A ACTIVE_PID_TO_IDX=()
declare -A ACTIVE_PGID_TO_IDX=()
declare -A TASK_NAME_MAP=()
declare -A TASK_GPUTOTAL_MAP=()
declare -A TASK_CUDA_MAP=()
declare -A TASK_LOG_MAP=()
declare -A TASK_RC_FILE_MAP=()
declare -A TASK_WALL_FILE_MAP=()
declare -A TASK_DIR_MAP=()
declare -A TASK_ATTEMPT_MAP=()
declare -A TASK_PGID_MAP=()
declare -A NEXT_ATTEMPT_MAP=()
declare -A SHUTDOWN_PGIDS=()
INTERRUPTED=0

sleep_ms() {
    local ms="$1"
    local sec
    sec="$(awk -v v="$ms" 'BEGIN { printf "%.3f", v / 1000 }')"
    sleep "$sec"
}

log_scheduler() {
    local message="$1"
    local ts
    ts="$(date -Is)"
    echo "[$ts] $message" | tee -a "$SCHEDULER_LOG"
}

snapshot_active_task_pgids() {
    local root_pid
    local pid
    local pgid
    local children_file
    local position
    local child
    local -a queue=()
    local -a children=()
    local -A seen=()

    SHUTDOWN_PGIDS=()
    for root_pid in "${!ACTIVE_PID_TO_IDX[@]}"; do
        queue=("$root_pid")
        position=0
        while [ "$position" -lt "${#queue[@]}" ]; do
            pid="${queue[$position]}"
            position=$((position + 1))
            if [ -n "${seen[$pid]:-}" ]; then
                continue
            fi
            seen["$pid"]=1

            pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)"
            if [[ "$pgid" =~ ^[0-9]+$ ]] && [ "$pgid" -gt 1 ]; then
                SHUTDOWN_PGIDS["$pgid"]=1
            fi

            for children_file in /proc/"${pid}"/task/*/children; do
                children=()
                if [ -r "$children_file" ]; then
                    read -r -a children < "$children_file" || true
                    for child in "${children[@]}"; do
                        queue+=("$child")
                    done
                fi
            done
        done
    done

    # Preserve the recorded task groups if a wrapper exits during discovery.
    for pgid in "${!ACTIVE_PGID_TO_IDX[@]}"; do
        if [[ "$pgid" =~ ^[0-9]+$ ]] && [ "$pgid" -gt 1 ]; then
            SHUTDOWN_PGIDS["$pgid"]=1
        fi
    done
}

signal_shutdown_targets() {
    local signal="${1:-TERM}"
    local pid
    local pgid

    for pgid in "${!SHUTDOWN_PGIDS[@]}"; do
        if [ -n "$pgid" ] && kill -0 "-$pgid" 2>/dev/null; then
            kill "-${signal}" "-$pgid" 2>/dev/null || true
        fi
    done

    for pid in "${!ACTIVE_PID_TO_IDX[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "-${signal}" "$pid" 2>/dev/null || true
        fi
    done
}

handle_interrupt() {
    local sig="${1:-INT}"
    local pid

    if [ "$INTERRUPTED" -eq 1 ]; then
        # 第二次 Ctrl+C 直接强杀，避免卡住。
        signal_shutdown_targets KILL
        exit 130
    fi
    INTERRUPTED=1

    log_scheduler "[中断] 捕获到 ${sig}，正在停止 ${#ACTIVE_PID_TO_IDX[@]} 个运行中任务..."
    snapshot_active_task_pgids
    signal_shutdown_targets TERM
    sleep 1
    signal_shutdown_targets KILL

    # 回收子进程，避免僵尸进程和残留后台任务。
    for pid in "${!ACTIVE_PID_TO_IDX[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    ACTIVE_PID_TO_IDX=()
    ACTIVE_PGID_TO_IDX=()
    log_scheduler "[中断] 所有任务已停止，退出调度器。"
    exit 130
}

trap 'handle_interrupt INT' INT
trap 'handle_interrupt TERM' TERM

refresh_runtime_not_use_config() {
    local reserved_list=()
    local gid
    for gid in "${!RESERVED_GPU_SET[@]}"; do
        reserved_list+=("$gid")
    done

    python3 - "$BASE_NOT_USE_CONFIG" "$RUNTIME_NOT_USE_CONFIG" "${#STATIC_NEVER_USE[@]}" "${STATIC_NEVER_USE[@]}" "${reserved_list[@]}" <<'PY'
import sys
from pathlib import Path

import yaml

base_config = sys.argv[1]
out_path = Path(sys.argv[2])
static_count = int(sys.argv[3])
static_not_use = [int(x) for x in sys.argv[4:4 + static_count]]
reserved = [int(x) for x in sys.argv[4 + static_count:]]
merged = []

if base_config:
    with open(base_config, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or []
    if not isinstance(loaded, list):
        raise SystemExit("not_use_config 必须是 GPU ID 列表")
    merged.extend(int(x) for x in loaded)

merged.extend(static_not_use)
merged.extend(reserved)
merged = sorted(set(merged))

with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(merged, f, sort_keys=False)
PY
}

reserve_gpus() {
    local cuda_ids="$1"
    local gid
    IFS=',' read -r -a _reserve_ids <<< "$cuda_ids"
    for gid in "${_reserve_ids[@]}"; do
        [ -n "$gid" ] && RESERVED_GPU_SET["$gid"]=1
    done
}

release_gpus() {
    local cuda_ids="$1"
    local gid
    IFS=',' read -r -a _release_ids <<< "$cuda_ids"
    for gid in "${_release_ids[@]}"; do
        [ -n "$gid" ] && unset 'RESERVED_GPU_SET[$gid]'
    done
}

acquire_gpus_nonblocking() {
    local needed="$1"
    local args=("$needed" "--nowait")

    refresh_runtime_not_use_config
    if [ -n "$BASE_ONLY_USE" ]; then
        args+=("--only-use" "$BASE_ONLY_USE")
    fi
    if [ -f "$RUNTIME_NOT_USE_CONFIG" ]; then
        args+=("--not-use-config" "$RUNTIME_NOT_USE_CONFIG")
    fi
    if [ "${#GPU_PASSTHROUGH_ARGS[@]}" -gt 0 ]; then
        args+=("${GPU_PASSTHROUGH_ARGS[@]}")
    fi

    get_free_gpus "${args[@]}" || true
}

run_task_background() {
    local task_dir="$1"
    local name="$2"
    local workdir="$3"
    local cuda_ids="$4"
    local log_file="$5"
    local rc_file="$6"
    local wall_file="$7"

    setsid "${SCRIPT_DIR}/gpu_job_runner_task.sh" \
        "$task_dir" "$name" "$workdir" "$log_file" "$rc_file" "$wall_file" "$cuda_ids" &
}

record_finished_task() {
    local pid="$1"
    local idx="${ACTIVE_PID_TO_IDX[$pid]}"
    local pgid
    local name="${TASK_NAME_MAP[$idx]}"
    local gpus="${TASK_GPUTOTAL_MAP[$idx]}"
    local cuda_ids="${TASK_CUDA_MAP[$idx]}"
    local log_file="${TASK_LOG_MAP[$idx]}"
    local rc_file="${TASK_RC_FILE_MAP[$idx]}"
    local wall_file="${TASK_WALL_FILE_MAP[$idx]}"
    local task_dir="${TASK_DIR_MAP[$idx]}"
    local attempt="${TASK_ATTEMPT_MAP[$idx]}"
    local rc wall

    set +e
    wait "$pid"
    rc="$?"
    set -e

    if [ -f "$rc_file" ]; then
        rc="$(tr -d '\r\n' < "$rc_file")"
    fi
    if [ -f "$wall_file" ]; then
        wall="$(tr -d '\r\n' < "$wall_file")"
    else
        wall=0
    fi

    echo -e "${idx}\t${name}\t${attempt}\t${gpus}\t${cuda_ids}\t${rc}\t${log_file}\t${wall}" >>"$SUMMARY"
    echo -e "${idx}\t${name}\t${attempt}\t${rc}" >>"$RESULTS"

    release_gpus "$cuda_ids"
    if [ "$rc" != "0" ]; then
        TASK_DIRS+=( "$task_dir" )
        total_tasks="${#TASK_DIRS[@]}"
        NEXT_ATTEMPT_MAP["$idx"]=$((attempt + 1))
        log_scheduler "[失败重排] [$idx] $name attempt=${attempt} exit=$rc，已加入队列末尾；gpus=${cuda_ids:-<none>} log=$log_file"
    else
        log_scheduler "[完成] [$idx] $name attempt=${attempt} exit=$rc gpus=${cuda_ids:-<none>} log=$log_file"
    fi

    pgid="${TASK_PGID_MAP[$idx]:-}"
    unset 'ACTIVE_PID_TO_IDX[$pid]'
    if [ -n "$pgid" ]; then
        unset 'ACTIVE_PGID_TO_IDX[$pgid]'
    fi
    unset 'TASK_PGID_MAP[$idx]'
}

poll_finished_tasks() {
    local finished=0
    local pid
    for pid in "${!ACTIVE_PID_TO_IDX[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi
        record_finished_task "$pid"
        finished=1
    done

    if [ "$finished" -eq 1 ] && [ "${sleep_after_ms}" -gt 0 ] 2>/dev/null; then
        sleep_ms "$sleep_after_ms"
    fi
}

shopt -s nullglob
TASK_DIRS=( "${STAGING}/"[0-9][0-9][0-9]_*/ )
shopt -u nullglob

next_task_pos=0
total_tasks="${#TASK_DIRS[@]}"
last_wait_log_ms=0
last_wait_idx=""

log_scheduler "调度开始。日志目录: $RUN_LOG_DIR"
if [ "$max_parallel_tasks" -gt 0 ]; then
    log_scheduler "并发上限: max_parallel_tasks=$max_parallel_tasks"
else
    log_scheduler "并发上限: 不限制（仅受 GPU 可用量约束）"
fi

while :; do
    poll_finished_tasks

    launched_any=0
    while [ "$next_task_pos" -lt "$total_tasks" ]; do
        if [ "$max_parallel_tasks" -gt 0 ] && [ "${#ACTIVE_PID_TO_IDX[@]}" -ge "$max_parallel_tasks" ]; then
            break
        fi

        task_dir="${TASK_DIRS[$next_task_pos]}"
        base="$(basename "$task_dir")"
        idx="${base%%_*}"
        name="$(tr -d '\r\n' < "${task_dir}/name")"
        gpus="$(tr -d '\r\n' < "${task_dir}/gpus")"
        workdir="$(tr -d '\r\n' < "${task_dir}/workdir")"
        attempt="${NEXT_ATTEMPT_MAP[$idx]:-1}"
        printf -v attempt_label "%03d" "$attempt"
        log_file="${RUN_LOG_DIR}/${idx}_${name}.attempt${attempt_label}.log"
        rc_file="${task_dir}/rc"
        wall_file="${task_dir}/wall"

        cuda_ids="$(acquire_gpus_nonblocking "$gpus")"
        if [ -z "$cuda_ids" ]; then
            now_ms="$(( $(date +%s) * 1000 ))"
            if [ "$idx" != "$last_wait_idx" ] || [ $((now_ms - last_wait_log_ms)) -ge "$wait_log_every_ms" ]; then
                log_scheduler "[等待] [$idx] $name 需要 ${gpus} 张卡，当前无足够空闲 GPU；已提交 ${next_task_pos}/${total_tasks}，运行中 ${#ACTIVE_PID_TO_IDX[@]}"
                last_wait_log_ms="$now_ms"
                last_wait_idx="$idx"
            fi
            break
        fi

        reserve_gpus "$cuda_ids"
        TASK_NAME_MAP["$idx"]="$name"
        TASK_GPUTOTAL_MAP["$idx"]="$gpus"
        TASK_CUDA_MAP["$idx"]="$cuda_ids"
        TASK_LOG_MAP["$idx"]="$log_file"
        TASK_RC_FILE_MAP["$idx"]="$rc_file"
        TASK_WALL_FILE_MAP["$idx"]="$wall_file"
        TASK_DIR_MAP["$idx"]="$task_dir"
        TASK_ATTEMPT_MAP["$idx"]="$attempt"

        log_scheduler "[提交] [$idx] $name attempt=${attempt} (需要 ${gpus} 张卡) -> CUDA_VISIBLE_DEVICES=$cuda_ids"
        run_task_background "$task_dir" "$name" "$workdir" "$cuda_ids" "$log_file" "$rc_file" "$wall_file"
        pid="$!"
        ACTIVE_PID_TO_IDX["$pid"]="$idx"
        pgid="$pid"
        ACTIVE_PGID_TO_IDX["$pgid"]="$idx"
        TASK_PGID_MAP["$idx"]="$pgid"
        next_task_pos=$((next_task_pos + 1))
        launched_any=1
        last_wait_idx=""
    done

    if [ "$next_task_pos" -ge "$total_tasks" ] && [ "${#ACTIVE_PID_TO_IDX[@]}" -eq 0 ]; then
        break
    fi

    if [ "$launched_any" -eq 0 ]; then
        sleep_ms "$scheduler_sleep_ms"
    fi
done

echo ""
echo "全部任务已调度完毕。汇总: $SUMMARY"
echo "退出码列表: $RESULTS"
echo "调度日志: $SCHEDULER_LOG"

# 失败任务会一直重试；能走到这里说明所有任务最终都成功了。
exit 0
