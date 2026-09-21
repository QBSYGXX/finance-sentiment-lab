"""Start a hidden local server, retaining logs and reporting its ready URL."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
from common import ROOT, save_json


# 模块：本地服务启动；核心业务：否。
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for port in range(8787, 8800):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) != 0:
                break
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                status = json.load(response)
            if status.get('application') == 'finance_sentiment_lab':
                print(f'Already ready: http://127.0.0.1:{port}', flush=True)
                return
        except Exception:
            pass
    else:
        raise RuntimeError('No free local port in 8787-8799')
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    with (logs / 'server_stdout.log').open('a', encoding='utf-8') as stdout, (logs / 'server_stderr.log').open('a', encoding='utf-8') as stderr:
        process = subprocess.Popen([sys.executable, '-u', '-X', 'utf8', str(ROOT / 'run.py'), 'serve', '--port', str(port)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    print(f'Server PID={process.pid}; waiting for /health', flush=True)
    for step in range(60):
        if process.poll() is not None:
            raise RuntimeError(f'Server exited {process.returncode}; see logs/server_stderr.log')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                result = json.load(response)
            if result.get('application') == 'finance_sentiment_lab':
                save_json(ROOT / '.runtime.json', {'pid': process.pid, 'port': port, 'url': f'http://127.0.0.1:{port}'})
                print(f'READY http://127.0.0.1:{port}', flush=True)
                return
        except Exception:
            pass
        if step and step % 10 == 0:
            print(f'Still loading models ({step}s); PID={process.pid}', flush=True)
        time.sleep(1)
    process.terminate()
    process.wait(timeout=10)
    raise TimeoutError('Server was not ready after 60 seconds')


if __name__ == '__main__':
    main()
