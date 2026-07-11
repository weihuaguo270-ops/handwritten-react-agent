@echo off
REM hagent — 全局命令入口
REM 将此文件所在目录添加到 PATH 即可全局使用
python "%~dp0hagent_cli.py" %*
