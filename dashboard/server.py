"""Agent 轨迹查看器（Flask）"""
from flask import Flask, jsonify, send_from_directory
import json
import os
import glob

app = Flask(__name__, static_folder='.', static_url_path='')

TRAJECTORIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trajectories')

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/api/trajectories')
def list_trajectories():
    if not os.path.exists(TRAJECTORIES_DIR):
        return jsonify([])
    files = sorted(glob.glob(os.path.join(TRAJECTORIES_DIR, '*.json')), key=os.path.getmtime, reverse=True)
    names = [os.path.basename(f) for f in files[:20]]
    return jsonify(names)

@app.route('/api/trajectories/<name>')
def get_trajectory(name):
    path = os.path.join(TRAJECTORIES_DIR, name)
    if not os.path.exists(path):
        return jsonify({'error': 'not found'}), 404
    with open(path, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=False)
