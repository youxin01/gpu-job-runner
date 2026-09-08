import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from get_free_gpus import GPURequest


def gpu(gpu_id, memory_util=0.0, load=0.0):
    return SimpleNamespace(id=gpu_id, memoryUtil=memory_util, load=load)


class GPURequestTests(unittest.TestCase):
    @patch("get_free_gpus.GPUtil.getGPUs")
    def test_selects_only_idle_allowed_gpus(self, get_gpus):
        get_gpus.return_value = [
            gpu(0),
            gpu(1, memory_util=0.5),
            gpu(2, load=0.5),
            gpu(3),
        ]

        requester = GPURequest(only_use=[0, 2, 3])

        self.assertEqual(requester.find_available_gpus(2, blocking=False), [0, 3])

    @patch("get_free_gpus.GPUtil.getGPUs")
    def test_applies_yaml_exclusion_file(self, get_gpus):
        get_gpus.return_value = [gpu(0), gpu(1), gpu(2)]
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "excluded.yaml"
            config.write_text("- 0\n- 2\n", encoding="utf-8")
            requester = GPURequest(not_use_config=str(config))

            self.assertEqual(requester.find_available_gpus(1, blocking=False), [1])

    @patch("get_free_gpus.GPUtil.getGPUs")
    def test_nonblocking_returns_none_without_capacity(self, get_gpus):
        get_gpus.return_value = [gpu(0, memory_util=0.2)]

        self.assertIsNone(GPURequest().find_available_gpus(1, blocking=False))

    @patch("get_free_gpus.GPUtil.getGPUs")
    def test_release_makes_gpu_available_again(self, get_gpus):
        get_gpus.return_value = [gpu(0)]
        requester = GPURequest()

        self.assertEqual(requester.find_available_gpus(1, blocking=False), [0])
        self.assertIsNone(requester.find_available_gpus(1, blocking=False))
        requester.release([0])
        self.assertEqual(requester.find_available_gpus(1, blocking=False), [0])


if __name__ == "__main__":
    unittest.main()
