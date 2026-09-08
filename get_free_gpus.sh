#!/bin/bash
# get_free_gpus.sh
#
# 提供 get_free_gpus shell 函数，用于获取空闲 GPU ID。
#
# 用法：
#   source get_free_gpus.sh
#   get_free_gpus 2                                    # 阻断模式：等待直到有 2 块空闲 GPU
#   get_free_gpus 2 --nowait                           # 非阻断模式：无空闲 GPU 则立即返回空
#   get_free_gpus 2 --not-use-config /path/to/a.yaml   # 指定排除 GPU 的配置文件
#   get_free_gpus 2 --only-use 0,1,2                   # 仅在 GPU 0,1,2 中查找
#
# 返回值：
#   成功时输出逗号分隔的 GPU ID（如 "0,3"），可通过 $(...) 捕获
#   失败时输出空字符串

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

get_free_gpus() {
    local num="$1"
    local extra_args=""

    if [ -z "$num" ]; then
        echo "用法: get_free_gpus <gpu数量> [--nowait] [--not-use-config <路径>] [--only-use <id列表>]" >&2
        return 1
    fi

    # 解析可选参数
    shift
    while [ $# -gt 0 ]; do
        case "$1" in
            --nowait|--non-blocking)
                extra_args="$extra_args --non-blocking"
                shift
                ;;
            --not-use-config)
                if [ -z "$2" ]; then
                    echo "错误: --not-use-config 需要指定配置文件路径" >&2
                    return 1
                fi
                extra_args="$extra_args --not-use-config $2"
                shift 2
                ;;
            --only-use)
                if [ -z "$2" ]; then
                    echo "错误: --only-use 需要指定 GPU ID 列表（逗号分隔）" >&2
                    return 1
                fi
                extra_args="$extra_args --only-use $2"
                shift 2
                ;;
            *)
                echo "未知参数: $1" >&2
                return 1
                ;;
        esac
    done
    python3 "${SCRIPT_DIR}/get_free_gpus.py" --num "$num" $extra_args
}
