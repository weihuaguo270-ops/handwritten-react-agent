"""启动前杀掉占用 5050 端口的旧进程"""
import subprocess
import sys
import os

port = 5050
if sys.platform.startswith('win'):
    cmd = f'for /f "tokens=5" %a in (\'netstat -ano ^| findstr ":{port}"\') do taskkill /F /PID %a 2>nul'
else:
    cmd = f'pkill -f "python.*server.py.*:{port}" 2>/dev/null'
subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
