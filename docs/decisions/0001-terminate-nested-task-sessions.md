# Decision: terminate nested task sessions

## Problem

GPU Job Runner promises to stop active task trees on `Ctrl+C` or `SIGTERM`, but it signals only the process group of each task wrapper. A legitimate task may use `setsid` to isolate a managed service such as vLLM. That service remains a descendant but has another process group, so the runner's TERM/KILL sequence can leave it consuming GPUs after the runner exits.

## Decision

Immediately before shutdown, the runner walks Linux `/proc` child relationships rooted at every active task wrapper, including children created by any thread, and snapshots every descendant process group. It sends TERM and then KILL to that stable set of groups while retaining direct PID signaling for recorded wrappers. Groups are deduplicated and invalid/system groups are excluded.

## Alternatives considered

- Forbid nested `setsid`: rejected because arbitrary YAML tasks may legitimately isolate services and the runner owns its documented cleanup contract.
- Merely extend the one-second grace period: rejected because it still cannot guarantee cleanup when an isolated child ignores TERM or outlives its parent.
- Require systemd/cgroup scopes: stronger containment, but rejected for this fix because it adds host-service and permission requirements to a standalone shell utility.

## Consequences

- Nested sessions remain available to YAML tasks without escaping runner interrupt cleanup.
- The one-second TERM grace period and exit status 130 remain unchanged.
- Discovery adds a small, shutdown-only `/proc` and `ps` traversal cost.
- A process that fully daemonizes and is reparented before the interrupt is outside the tracked task-tree guarantee.

## Verification

- `python3 -m unittest tests.test_runner.RunnerTests.test_ctrl_c_stops_descendant_in_nested_session -v` reproduces the leak before the fix and passes afterward.
- `conda run --no-capture-output -n statbench python -m unittest discover -s tests -v` passes all 9 repository tests, including ordinary task-tree cleanup and retry behavior.
- `bash -n get_free_gpus.sh gpu_job_runner.sh gpu_job_runner_task.sh` verifies shell syntax.
