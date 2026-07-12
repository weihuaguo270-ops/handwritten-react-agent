"""
Dashboard — Agent 执行轨迹可视化

从手写版 src/react_agent/tools/dashboard.py 迁移。
显示 Harness 记录的轨迹文件，支持回放和查看使用统计。

用法：
    python -m experiments.langgraph.graph.dashboard
"""
import json
import os
import glob
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime


TRAJECTORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "trajectories",
)
HOST = "127.0.0.1"
PORT = 8710


# ── API 处理器 ──

class DashboardHandler(SimpleHTTPRequestHandler):
    """Dashboard 请求处理器"""

    def do_GET(self):
        if self.path == "/api/trajectories":
            self._send_json(self._list_trajectories())
        elif self.path.startswith("/api/trajectory/"):
            name = self.path.split("/api/trajectory/")[1]
            self._send_json(self._load_trajectory(name))
        elif self.path == "/api/stats":
            self._send_json(self._get_stats())
        elif self.path == "/":
            self._send_html()
        else:
            super().do_GET()

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode())

    def _send_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        html = self._build_html()
        self.wfile.write(html.encode())

    def _list_trajectories(self):
        """列出所有轨迹文件"""
        files = sorted(glob.glob(os.path.join(TRAJECTORY_DIR, "traj_*.json")),
                       key=os.path.getmtime, reverse=True)
        items = []
        for f in files[:50]:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                items.append({
                    "name": os.path.basename(f),
                    "query": data.get("query", "")[:80],
                    "time": datetime.fromtimestamp(os.path.getmtime(f)).isoformat(),
                    "steps": len(data.get("steps", [])),
                    "model": data.get("model", ""),
                })
            except Exception:
                pass
        return items

    def _load_trajectory(self, name: str):
        """加载单个轨迹"""
        path = os.path.join(TRAJECTORY_DIR, name)
        if not os.path.exists(path):
            return {"error": "not found"}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_stats(self):
        """统计信息"""
        files = glob.glob(os.path.join(TRAJECTORY_DIR, "traj_*.json"))
        total = len(files)
        tool_counts = {}
        total_steps = 0
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                steps = data.get("steps", [])
                total_steps += len(steps)
                for step in steps:
                    if isinstance(step, dict) and step.get("type") == "action":
                        name = step.get("name", "unknown")
                        tool_counts[name] = tool_counts.get(name, 0) + 1
            except Exception:
                pass
        return {
            "total_trajectories": total,
            "total_steps": total_steps,
            "avg_steps": round(total_steps / total, 1) if total else 0,
            "tool_usage": dict(sorted(tool_counts.items(),
                                      key=lambda x: x[1], reverse=True)),
        }

    def _build_html(self) -> str:
        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Agent Dashboard — 轨迹浏览器</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 20px auto; padding: 0 20px; background: #f5f5f5; }}
  h1 {{ color: #333; }}
  .card {{ background: #fff; border-radius: 8px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stat {{ display: inline-block; margin: 0 20px 10px 0; }}
  .stat-value {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
  .stat-label {{ font-size: 12px; color: #666; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; font-size: 14px; }}
  th {{ color: #666; font-weight: 500; }}
  tr:hover {{ background: #f0f7ff; cursor: pointer; }}
  .step {{ margin: 8px 0; padding: 8px 12px; border-left: 3px solid #ddd; background: #fafafa; }}
  .step-thought {{ border-color: #2563eb; }}
  .step-action {{ border-color: #eab308; }}
  .step-observation {{ border-color: #22c55e; }}
</style>
</head>
<body>
<h1>🤖 Agent Dashboard</h1>
<div class="card" id="stats"></div>
<div class="card">
  <h3>轨迹列表</h3>
  <table id="traj-list"><tr><th>时间</th><th>查询</th><th>步数</th><th>模型</th></tr></table>
</div>
<div class="card" id="detail" style="display:none">
  <h3 id="detail-title">轨迹详情</h3>
  <div id="detail-content"></div>
</div>
<script>
async function load() {{
  const [trajs, stats] = await Promise.all([
    fetch('/api/trajectories').then(r=>r.json()),
    fetch('/api/stats').then(r=>r.json()),
  ]);
  document.getElementById('stats').innerHTML = `
    <div class="stat"><div class="stat-value">${{stats.total_trajectories}}</div><div class="stat-label">轨迹总数</div></div>
    <div class="stat"><div class="stat-value">${{stats.total_steps}}</div><div class="stat-label">总步数</div></div>
    <div class="stat"><div class="stat-value">${{stats.avg_steps}}</div><div class="stat-label">平均步数</div></div>
  `;
  const tbody = document.getElementById('traj-list');
  trajs.forEach(t => {{
    const row = tbody.insertRow();
    row.onclick = () => showDetail(t.name);
    row.innerHTML = `<td>${{t.time.slice(0,16)}}</td><td>${{t.query}}</td><td>${{t.steps}}</td><td>${{t.model}}</td>`;
  }});
}}
async function showDetail(name) {{
  const data = await fetch('/api/trajectory/'+name).then(r=>r.json());
  const div = document.getElementById('detail');
  div.style.display = 'block';
  document.getElementById('detail-title').textContent = '轨迹: '+name;
  document.getElementById('detail-content').innerHTML = (data.steps||[]).map(s => `
    <div class="step step-${{s.type||'unknown'}}">
      <strong>${{s.type||'步骤'}}:</strong> ${{s.content||s.name||''}}
      <br><small>${{s.args ? JSON.stringify(s.args).slice(0,200) : ''}}</small>
    </div>
  `).join('');
}}
load();
</script>
</body>
</html>"""
        # 注意：这里用了 Python 的 f-string，大括号转义为 {{ }}


def main():
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    server = HTTPServer((HOST, PORT), DashboardHandler)
    print(f"\n🤖 Agent Dashboard — 轨迹浏览器")
    print(f"   地址: http://{HOST}:{PORT}")
    print(f"   轨迹目录: {TRAJECTORY_DIR}")
    print(f"   按 Ctrl+C 退出\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard 已停止。")


if __name__ == "__main__":
    main()
