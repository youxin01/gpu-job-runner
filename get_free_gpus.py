#!/usr/bin/env python3
"""
获取空闲 GPU ID 的命令行工具。

用法：
    python get_free_gpus.py --num 2 [--non-blocking] [--not-use-config path/to/config.yaml]

返回：
    成功时输出逗号分隔的 GPU ID（如 "0,3"）
    失败（非阻断模式下无空闲 GPU）时输出空字符串
"""

import argparse
import sys
import time
from typing import Optional, Tuple, List

import GPUtil
from yaml import safe_load


class GPURequest:
    """独立 GPU 请求类，用于查找并分配空闲 GPU。"""

    def __init__(self, only_use: Optional[List[int]] = None, not_use_config: Optional[str] = None):
        """
        Parameters
        ----------
        only_use : Optional[List[int]]
            仅允许使用的 GPU ID 列表。为 None 时表示所有 GPU 均可使用。
        not_use_config : Optional[str]
            不使用的 GPU 配置文件路径（yaml 格式，内容为 GPU ID 列表）。
            为 None 时表示没有排除的 GPU，所有 GPU 均可使用。
        """
        self.have_use: set = set()
        self.not_use: set = set()
        self.only_use = only_use
        self.not_use_config = not_use_config

    def _get_not_use(self):
        """从配置文件中读取不使用的 GPU ID 列表。"""
        try:
            if self.not_use_config is not None:
                with open(self.not_use_config, encoding="utf-8") as config_file:
                    data = safe_load(config_file)
                self.not_use = set(data) if data else set()
            else:
                self.not_use = set()
        except Exception as e:
            print(f"[GPURequest] 读取 not_use_config 失败: {e}", file=sys.stderr)
            time.sleep(1)

    def find_available_gpus(
        self, blocks_needed: int, blocking: bool = True
    ) -> Optional[List[int]]:
        """
        根据需要的 GPU 数量，返回可用的 GPU ID 列表。

        Parameters
        ----------
        blocks_needed : int
            需要的 GPU 数量。设为 0 时返回空列表。
        blocking : bool
            True 为阻断模式，会一直等待直到有空闲 GPU；
            False 为非阻断模式，无空闲 GPU 时立即返回 None。

        Returns
        -------
        Optional[List[int]]
            可用的 GPU ID 列表，或 None（非阻断模式下无空闲 GPU 时）。
        """
        if blocks_needed == 0:
            return []

        while True:
            self._get_not_use()
            gpus = GPUtil.getGPUs()

            available_gpus = [
                gpu.id for gpu in gpus
                if gpu.memoryUtil < 0.1
                and gpu.load < 0.1
                and gpu.id not in self.have_use
                and gpu.id not in self.not_use
                and (self.only_use is None or int(gpu.id) in self.only_use)
            ]

            if len(available_gpus) >= blocks_needed:
                give = available_gpus[:blocks_needed]
                for i in give:
                    self.have_use.add(int(i))
                return give

            if blocking:
                time.sleep(1)
            else:
                return None

    def release(self, gpu_ids: List[int]):
        """释放已使用的 GPU，使其重新变为可用状态。"""
        for gid in gpu_ids:
            try:
                self.have_use.remove(int(gid))
            except KeyError:
                print(f"[GPURequest] GPU {gid} 不在已使用集合中: {self.have_use}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="获取指定数量的空闲 GPU ID")
    parser.add_argument("--num", "-n", type=int, required=True, help="需要的 GPU 数量")
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        default=False,
        help="非阻断模式：无空闲 GPU 时立即返回空（默认为阻断模式，会一直等待）",
    )
    parser.add_argument(
        "--not-use-config",
        type=str,
        default=None,
        help="不使用的 GPU 配置文件路径（yaml 格式），不指定则所有 GPU 均可使用",
    )
    parser.add_argument(
        "--only-use",
        type=str,
        default=None,
        help="仅使用的 GPU ID，逗号分隔（如 '0,1,2'），不指定则所有 GPU 均可使用",
    )
    args = parser.parse_args()

    only_use = None
    if args.only_use:
        only_use = [int(x) for x in args.only_use.split(",")]

    requester = GPURequest(only_use=only_use, not_use_config=args.not_use_config)
    gpus = requester.find_available_gpus(blocks_needed=args.num, blocking=not args.non_blocking)

    if gpus is not None:
        print(",".join(str(g) for g in gpus))
    else:
        print("")


if __name__ == "__main__":
    main()
