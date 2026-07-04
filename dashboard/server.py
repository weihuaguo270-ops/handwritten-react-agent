"""Agent 轨迹查看器（Flask）"""
from flask import Flask, jsonify, send_from_directory
import json
import os
import glob
import time

app = Flask(__name__, static_folder='.', static_url_path='')

TRAJECTORIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trajectories')


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


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=False)
