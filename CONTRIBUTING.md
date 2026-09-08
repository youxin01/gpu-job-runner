# Contributing

Contributions are welcome through GitHub issues and pull requests.

Run the local checks before submitting a change:

```bash
python3 -m unittest discover -s tests -v
bash -n get_free_gpus.sh gpu_job_runner.sh gpu_job_runner_task.sh
python3 -m py_compile get_free_gpus.py gpu_job_runner_parse.py
```

Keep the root script names, the `Ctrl+C` task-tree termination behavior, and all
existing YAML fields backward compatible. New behavior should be optional unless
it fixes a confirmed defect.
