"""Agent 对话 + 轨迹查看器（Flask）"""
from flask import Flask, jsonify, send_from_directory, request
import json
import os
import glob
import time
import sys
import subprocess

app = Flask(__name__, static_folder='.', static_url_path='')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJECTORIES_DIR = os.path.join(BASE_DIR, 'trajectories')
REACT_LOOP = os.path.join(BASE_DIR, 'react_loop.py')


@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')


@app.route('/api/trajectories')
def list_trajectories():
    if not os.path.exists(TRAJECTORIES_DIR):
        return jsonify([])
    files = sorted(glob.glob(os.path.join(TRAJECTORIES_DIR, '*.json')),
                   key=os.path.getmtime, reverse=True)
    result = []
    for f in files[:50]:
        try:
            with open(f, encoding='utf-8') as fh:
                data = json.load(fh)
            result.append({
                'name': os.path.basename(f),
                'query': data.get('query', ''),
                'model': data.get('model', ''),
                'steps': data.get('total_steps', 0),
                'duration': data.get('total_duration_seconds', 0),
                'tokens': data.get('total_tokens_estimated', 0),
                'timestamp': data.get('timestamp', ''),
                'session_id': data.get('session_id', ''),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return jsonify(result)


@app.route('/api/trajectories/<name>')
def get_trajectory(name):
    path = os.path.join(TRAJECTORIES_DIR, name)
    if not os.path.exists(path):
        return jsonify({'error': 'not found'}), 404
    with open(path, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))


@app.route('/api/chat', methods=['POST'])
def chat():
    """接收用户 query，调用 react_loop.py 执行，返回轨迹"""
    data = request.get_json(force=True)
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'query 不能为空'}), 400

    try:
        result = subprocess.run(
            [sys.executable, REACT_LOOP, query],
            capture_output=True, text=True, timeout=60,
            cwd=BASE_DIR,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()[:500] if result.stderr else ''

        # 查找最新轨迹文件
        latest_traj = None
        if os.path.exists(TRAJECTORIES_DIR):
            files = sorted(glob.glob(os.path.join(TRAJECTORIES_DIR, '*.json')),
                           key=os.path.getmtime, reverse=True)
            if files:
                with open(files[0], encoding='utf-8') as fh:
                    latest_traj = json.load(fh)

        return jsonify({
            'stdout': stdout,
            'stderr': stderr,
            'trajectory': latest_traj,
            'exit_code': result.returncode,
        })

    except subprocess.TimeoutExpired:
        return jsonify({'error': '执行超时（60秒）', 'stdout': '', 'stderr': '', 'trajectory': None}), 504
    except Exception as e:
        return jsonify({'error': str(e), 'stdout': '', 'stderr': '', 'trajectory': None}), 500


@app.route('/api/trajectories/clear', methods=['POST'])
def clear_trajectories():
    """删除轨迹文件"""
    data = request.get_json(force=True)
    days = data.get('days', 0)
    if not os.path.exists(TRAJECTORIES_DIR):
        return jsonify({'message': '没有轨迹文件可删除'})

    pattern = os.path.join(TRAJECTORIES_DIR, 'traj_*.json')
    files = glob.glob(pattern)
    if not files:
        return jsonify({'message': '没有轨迹文件可删除'})

    now = time.time()
    cutoff = now - days * 86400 if days > 0 else now + 1
    removed = 0
    for f in files:
        if days > 0:
            mtime = os.path.getmtime(f)
            if mtime >= cutoff:
                continue
        try:
            os.remove(f)
            removed += 1
        except OSError:
            continue
    msg = f'已删除 {removed} 个轨迹文件' if removed else '没有符合条件的轨迹文件'
    return jsonify({'message': msg})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=False)
