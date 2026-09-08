import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER = ROOT / "gpu_job_runner_parse.py"


class ParserTests(unittest.TestCase):
    def test_parses_global_and_task_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config = temp / "jobs.yaml"
            staging = temp / "staging"
            config.write_text(
                """
log_dir: /tmp/example-logs
max_parallel_tasks: 2
never_use_gpus: [4, 5]
get_free_gpus_extra: [--only-use, "0,1,2,3"]
tasks:
  - name: demo/task
    gpus: 2
    workdir: /tmp
    cmd: |
      echo "$CUDA_VISIBLE_DEVICES"
""".lstrip(),
                encoding="utf-8",
            )

            subprocess.run(
                [sys.executable, str(PARSER), str(config), str(staging)],
                check=True,
                capture_output=True,
                text=True,
            )

            task_dir = staging / "000_demo_task"
            self.assertEqual((staging / "_global.max_parallel_tasks").read_text().strip(), "2")
            self.assertEqual((staging / "_global.never_use_gpus").read_text().split(), ["4", "5"])
            self.assertEqual((task_dir / "gpus").read_text().strip(), "2")
            self.assertEqual((task_dir / "workdir").read_text().strip(), "/tmp")
            self.assertIn('echo "$CUDA_VISIBLE_DEVICES"', (task_dir / "cmd.sh").read_text())

    def test_rejects_task_without_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config = temp / "jobs.yaml"
            config.write_text("tasks:\n  - name: invalid\n    gpus: 1\n", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(PARSER), str(config), str(temp / "staging")],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("缺少 gpus 或 cmd", result.stderr)


if __name__ == "__main__":
    unittest.main()
