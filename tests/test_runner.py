import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "gpu_job_runner.sh"


def write_fake_gpu_helper(path):
    path.write_text(
        """#!/usr/bin/env bash
get_free_gpus() {
    echo 0
}
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def runner_env():
    env = os.environ.copy()
    env["GPU_JOB_SCHEDULER_SLEEP_MS"] = "20"
    env["GPU_JOB_WAIT_LOG_EVERY_MS"] = "20"
    return env


def process_is_running(pid):
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return False
    fields = stat_path.read_text(encoding="utf-8").split()
    return len(fields) > 2 and fields[2] not in {"Z", "X"}


class RunnerTests(unittest.TestCase):
    def test_requeues_failed_task_and_preserves_attempt_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            helper = temp / "fake_gpu_helper.sh"
            marker = temp / "first_attempt"
            logs = temp / "logs"
            config = temp / "jobs.yaml"
            write_fake_gpu_helper(helper)
            config.write_text(
                f"""
log_dir: {logs}
max_parallel_tasks: 1
tasks:
  - name: retry_once
    gpus: 1
    cmd: |
      if [ ! -f {marker} ]; then
        touch {marker}
        exit 3
      fi
      echo recovered
""".lstrip(),
                encoding="utf-8",
            )

            result = subprocess.run(
                ["bash", str(RUNNER), str(config), "--gpu-helper", str(helper)],
                env=runner_env(),
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = (logs / "summary.tsv").read_text(encoding="utf-8")
            self.assertIn("\t1\t1\t0\t3\t", summary)
            self.assertIn("\t2\t1\t0\t0\t", summary)
            self.assertTrue((logs / "000_retry_once.attempt001.log").is_file())
            self.assertIn("recovered", (logs / "000_retry_once.attempt002.log").read_text())

    def test_ctrl_c_stops_active_task_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            helper = temp / "fake_gpu_helper.sh"
            started = temp / "started"
            child_started = temp / "child_started"
            logs = temp / "logs"
            config = temp / "jobs.yaml"
            write_fake_gpu_helper(helper)
            config.write_text(
                f"""
log_dir: {logs}
tasks:
  - name: long_task
    gpus: 1
    cmd: |
      echo $$ > {started}
      sleep 60 &
      echo $! > {child_started}
      wait
""".lstrip(),
                encoding="utf-8",
            )

            process = subprocess.Popen(
                ["bash", str(RUNNER), str(config), "--gpu-helper", str(helper)],
                env=runner_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                deadline = time.time() + 5
                while (not started.exists() or not child_started.exists()) and time.time() < deadline:
                    time.sleep(0.05)
                self.assertTrue(started.exists(), "task did not start")
                self.assertTrue(child_started.exists(), "child process did not start")
                task_pids = [
                    int(started.read_text().strip()),
                    int(child_started.read_text().strip()),
                ]

                os.kill(process.pid, signal.SIGINT)
                process.wait(timeout=5)
                self.assertEqual(process.returncode, 130)

                deadline = time.time() + 2
                while time.time() < deadline:
                    if not any(process_is_running(pid) for pid in task_pids):
                        break
                    time.sleep(0.05)
                else:
                    survivors = [pid for pid in task_pids if process_is_running(pid)]
                    self.fail(f"task processes survived Ctrl+C: {survivors}")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
                if process.stdout is not None:
                    process.stdout.close()


if __name__ == "__main__":
    unittest.main()
