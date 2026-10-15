# Decision: support tasks that need no GPU

## Problem

A job queue can include CPU work or calls to a remote API. Previously, a task
requesting zero GPUs waited indefinitely because an empty GPU assignment was
treated as insufficient capacity. The task wrapper also rejected an empty GPU
assignment.

## Decision

`gpus: 0` is a valid task request. The scheduler accepts an empty assignment for
that request, and the wrapper exports an empty `CUDA_VISIBLE_DEVICES`. Negative
GPU counts are rejected by the parser. Existing positive-GPU requests keep their
allocation behavior. Zero-GPU tasks use the same concurrency limit, retry policy,
logs, and interrupt cleanup as other tasks.

## Alternatives considered

Running CPU or API work outside the runner avoids changing GPU allocation, but
loses the shared retry, concurrency, logging, and cleanup behavior. Reusing the
existing GPU-count field lets those tasks use the same queue format.

## Consequences

No new YAML field is needed. A zero-GPU task receives no visible CUDA devices,
but this is an environment setting rather than hardware access isolation.
`max_parallel_tasks: 0` imposes no concurrency cap on zero-GPU tasks; callers can
set a finite value to bound CPU or API load. The queue still attempts tasks in
order, so an earlier task waiting for GPUs can delay a later zero-GPU task.

## Verification

- `python3 -m unittest discover -s tests -p test_runner.py -v` includes a
  zero-GPU task whose helper returns no devices. It checks completion, an empty
  `CUDA_VISIBLE_DEVICES`, the task log, and the summary row.
- `python3 -m unittest discover -s tests -p test_parser.py -v` checks that a
  negative GPU count fails with a configuration error.
- Both README files document `gpus: 0`, concurrency limits, and examples.
