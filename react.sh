#!/usr/bin/env bash
# react — 全局命令入口 (bash/PowerShell)
# 将此文件所在目录添加到 PATH 即可全局使用
cd "$(dirname "$0")" && python react_cli.py "$@"
