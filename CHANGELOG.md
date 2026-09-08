# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Fix `Ctrl+C` cleanup for task descendants that create nested sessions with
  `setsid`, preventing managed services such as vLLM from surviving the runner.

## 1.0.0 - 2026-09-08

- Publish the YAML-driven GPU job runner and standalone idle-GPU selector.
- Preserve automatic retry, concurrency limits, GPU allowlists, and exclusions.
- Isolate task process groups so `Ctrl+C` terminates complete task trees.
- Add English and Chinese documentation, examples, tests, and CI.
