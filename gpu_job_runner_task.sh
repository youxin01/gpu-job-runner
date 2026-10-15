#!/usr/bin/env bash
# Internal task wrapper. gpu_job_runner.sh starts this script in a dedicated
# session so one interrupt can terminate the complete task process tree.
set -uo pipefail

task_dir="${1:?missing task directory}"
name="${2:?missing task name}"
workdir="$3"
log_file="${4:?missing log file}"
rc_file="${5:?missing exit-code file}"
wall_file="${6:?missing wall-time file}"
cuda_ids="${7-}"

export CUDA_VISIBLE_DEVICES="$cuda_ids"
: >"$log_file"

echo "---- $(date -Is) START $name ----" | tee -a "$log_file"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$log_file"
echo "cmd: ${task_dir}/cmd.sh" | tee -a "$log_file"
echo "" | tee -a "$log_file"

t0="$(date +%s)"
rc=0
if [ -n "$workdir" ]; then
    if [ ! -d "$workdir" ]; then
        echo "工作目录不存在: $workdir" | tee -a "$log_file" >&2
        rc=1
    else
        pushd "$workdir" >/dev/null
        set +e
        bash "${task_dir}/cmd.sh" 2>&1 | tee -a "$log_file"
        rc="${PIPESTATUS[0]}"
        set -e
        popd >/dev/null
    fi
else
    set +e
    bash "${task_dir}/cmd.sh" 2>&1 | tee -a "$log_file"
    rc="${PIPESTATUS[0]}"
    set -e
fi
t1="$(date +%s)"

printf '%s\n' "$rc" >"$rc_file"
printf '%s\n' "$((t1 - t0))" >"$wall_file"
echo "" | tee -a "$log_file"
echo "---- $(date -Is) END $name exit=$rc wall=$((t1 - t0))s ----" | tee -a "$log_file"
exit "$rc"
