#!/usr/bin/env python3
"""Simple web server to visualize simulation results.

Usage:
    python visualize_server.py [port]

Then open http://localhost:8080 in your browser.
"""

import os
import json
import http.server
import socketserver
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PORT = 8080
RESULTS_DIR = Path("distributed_results")

WEATHERS = ["cold_winter", "mild_winter", "temperate", "mild_summer", "hot_summer", "tropical"]
POLICIES = ["all_off", "all_on", "bang_bang"]

WEATHER_NAMES = {
    "cold_winter": "Cold Winter", "mild_winter": "Mild Winter", "temperate": "Temperate",
    "mild_summer": "Mild Summer", "hot_summer": "Hot Summer", "tropical": "Tropical"
}
POLICY_NAMES = {"all_off": "All Off", "all_on": "All On", "bang_bang": "Bang-Bang"}


def get_buildings():
    if not RESULTS_DIR.exists():
        return []
    return sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()])


def load_summaries(building):
    summaries = []
    for weather in WEATHERS:
        for policy in POLICIES:
            path = RESULTS_DIR / building / weather / policy / "summary.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    data['weather'] = weather
                    data['policy'] = policy
                    summaries.append(data)
    return summaries


def generate_html(building):
    summaries = load_summaries(building)
    buildings = get_buildings()

    # Build rewards table
    rewards = {}
    for s in summaries:
        rewards[(s['policy'], s['weather'])] = s['total_reward']

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Smart Buildings Visualizer - {building}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .nav {{ margin-bottom: 20px; }}
        .nav a {{ margin-right: 10px; padding: 8px 16px; background: #4a90d9; color: white; text-decoration: none; border-radius: 4px; }}
        .nav a:hover {{ background: #357abd; }}
        .nav a.active {{ background: #2d6da3; }}
        .section {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
        th {{ background: #4a90d9; color: white; }}
        .policy-header {{ background: #5ba55b; color: white; font-weight: bold; }}
        .reward {{ font-weight: bold; }}
        .reward-high {{ background: #c8e6c9; }}
        .reward-mid {{ background: #fff9c4; }}
        .reward-low {{ background: #ffcdd2; }}
        .gif-grid {{ display: grid; grid-template-columns: auto repeat(6, 1fr); gap: 10px; align-items: start; }}
        .gif-grid img {{ width: 100%; max-width: 200px; border-radius: 4px; cursor: pointer; transition: transform 0.2s; }}
        .gif-grid img:hover {{ transform: scale(1.05); }}
        .gif-cell {{ text-align: center; }}
        .header-cell {{ font-weight: bold; padding: 10px; }}
        .weather-header {{ background: #4a90d9; color: white; border-radius: 4px; }}
        .policy-cell {{ background: #5ba55b; color: white; border-radius: 4px; writing-mode: vertical-rl; text-orientation: mixed; padding: 20px 10px; }}
        .pending {{ color: #999; font-style: italic; }}
        .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; justify-content: center; align-items: center; }}
        .modal img {{ max-width: 90%; max-height: 90%; }}
        .modal.active {{ display: flex; }}
    </style>
</head>
<body>
    <h1>Smart Buildings Simulation Results</h1>

    <div class="nav">
'''

    for b in buildings:
        active = "active" if b == building else ""
        html += f'        <a href="/?building={b}" class="{active}">{b}</a>\n'

    html += '''    </div>

    <div class="section">
        <h2>Rewards Heatmap</h2>
        <table>
            <tr><th></th>'''

    for w in WEATHERS:
        html += f'<th>{WEATHER_NAMES[w]}</th>'
    html += '</tr>\n'

    # Calculate min/max for coloring
    if rewards:
        min_r = min(rewards.values())
        max_r = max(rewards.values())
        range_r = max_r - min_r if max_r != min_r else 1

    for policy in POLICIES:
        html += f'        <tr><td class="policy-header">{POLICY_NAMES[policy]}</td>'
        for weather in WEATHERS:
            key = (policy, weather)
            if key in rewards:
                r = rewards[key]
                # Color based on value
                pct = (r - min_r) / range_r if rewards else 0
                if pct > 0.66:
                    cls = "reward-high"
                elif pct > 0.33:
                    cls = "reward-mid"
                else:
                    cls = "reward-low"
                html += f'<td class="reward {cls}">{r:.1f}</td>'
            else:
                html += '<td class="pending">-</td>'
        html += '</tr>\n'

    html += '''        </table>
    </div>

    <div class="section">
        <h2>Simulation GIFs</h2>
        <p><em>Click any GIF to view larger</em></p>
        <div class="gif-grid">
            <div></div>
'''

    for w in WEATHERS:
        html += f'            <div class="header-cell weather-header">{WEATHER_NAMES[w]}</div>\n'

    for policy in POLICIES:
        html += f'            <div class="header-cell policy-cell">{POLICY_NAMES[policy]}</div>\n'
        for weather in WEATHERS:
            gif_path = RESULTS_DIR / building / weather / policy / "simulation.gif"
            png_path = RESULTS_DIR / building / weather / policy / "overview.png"
            if png_path.exists():
                # Show static PNG, click to view GIF
                png_src = f"/img/{building}/{weather}/{policy}/overview.png"
                gif_src = f"/gif/{building}/{weather}/{policy}/simulation.gif"
                html += f'            <div class="gif-cell"><img src="{png_src}" onclick="showModal(\'{gif_src}\')" title="Click for animation" loading="lazy"></div>\n'
            elif gif_path.exists():
                src = f"/gif/{building}/{weather}/{policy}/simulation.gif"
                html += f'            <div class="gif-cell"><img src="{src}" onclick="showModal(this.src)" loading="lazy"></div>\n'
            else:
                html += '            <div class="gif-cell pending">pending</div>\n'

    html += '''        </div>
    </div>

    <div class="modal" id="modal" onclick="hideModal()">
        <img id="modal-img" src="">
    </div>

    <script>
        function showModal(src) {
            document.getElementById('modal-img').src = src;
            document.getElementById('modal').classList.add('active');
        }
        function hideModal() {
            document.getElementById('modal').classList.remove('active');
        }
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') hideModal();
        });
    </script>
</body>
</html>'''

    return html


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        # Serve PNG files
        if parsed.path.startswith('/img/'):
            img_path = RESULTS_DIR / parsed.path[5:]  # Remove '/img/' prefix
            if img_path.exists() and img_path.suffix == '.png':
                file_size = img_path.stat().st_size
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.send_header('Content-Length', str(file_size))
                self.send_header('Cache-Control', 'max-age=86400')
                self.end_headers()
                with open(img_path, 'rb') as f:
                    self.wfile.write(f.read())
                return

        # Serve GIF files - stream in chunks
        if parsed.path.startswith('/gif/'):
            gif_path = RESULTS_DIR / parsed.path[5:]  # Remove '/gif/' prefix
            if gif_path.exists() and gif_path.suffix == '.gif':
                file_size = gif_path.stat().st_size
                self.send_response(200)
                self.send_header('Content-type', 'image/gif')
                self.send_header('Content-Length', str(file_size))
                self.send_header('Cache-Control', 'max-age=86400')  # Cache for 24 hours
                self.end_headers()
                # Stream in 64KB chunks
                with open(gif_path, 'rb') as f:
                    while chunk := f.read(65536):
                        self.wfile.write(chunk)
                return
            else:
                self.send_error(404)
                return

        # Serve main page
        if parsed.path == '/' or parsed.path.startswith('/?'):
            query = parse_qs(parsed.query)
            buildings = get_buildings()
            building = query.get('building', [buildings[0] if buildings else 'building_0002'])[0]

            html = generate_html(building)
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
            return

        # Default handler
        super().do_GET()


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving at http://localhost:{port}")
        print(f"Results directory: {RESULTS_DIR.absolute()}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server...")
