# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Add global YAML `max_retries` control while preserving unlimited retries by
  default, and return nonzero when any task exhausts its retry allowance.
- Fix `Ctrl+C` cleanup for task descendants that create nested sessions with
  `setsid`, preventing managed services such as vLLM from surviving the runner.
- Handle terminal/tmux `SIGHUP` during shutdown so it cannot interrupt cleanup
  after `Ctrl+C` and orphan active task trees.
- Add a guarded YAML-driven utility for clearing false-busy NVIDIA utilization
  state with a short-lived CUDA context.

## 1.0.0 - 2026-09-08

- Publish the YAML-driven GPU job runner and standalone idle-GPU selector.
- Preserve automatic retry, concurrency limits, GPU allowlists, and exclusions.
- Isolate task process groups so `Ctrl+C` terminates complete task trees.
- Add English and Chinese documentation, examples, tests, and CI.
