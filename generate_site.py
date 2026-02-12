#!/usr/bin/env python3
"""Generate a static GitHub Pages site from simulation results.

Usage:
    python generate_site.py [results_dir] [output_dir]

    # Default: reads from distributed_results/, outputs to docs/
    python generate_site.py

    # Then push to GitHub and enable Pages on the docs/ folder
"""

import os
import json
import shutil
import base64
from pathlib import Path
from argparse import ArgumentParser

WEATHERS = ["cold_winter", "mild_winter", "temperate", "mild_summer", "hot_summer", "tropical"]
POLICIES = ["all_off", "all_on", "bang_bang"]

WEATHER_NAMES = {
    "cold_winter": "Cold Winter", "mild_winter": "Mild Winter", "temperate": "Temperate",
    "mild_summer": "Mild Summer", "hot_summer": "Hot Summer", "tropical": "Tropical"
}
POLICY_NAMES = {"all_off": "All Off", "all_on": "All On", "bang_bang": "Bang-Bang"}


def get_buildings(results_dir):
    """Get list of buildings with results."""
    if not results_dir.exists():
        return []
    return sorted([d.name for d in results_dir.iterdir() if d.is_dir()])


def load_summaries(results_dir, building):
    """Load summary.json files for a building."""
    summaries = []
    for weather in WEATHERS:
        for policy in POLICIES:
            path = results_dir / building / weather / policy / "summary.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    data['weather'] = weather
                    data['policy'] = policy
                    summaries.append(data)
    return summaries


def generate_building_page(results_dir, output_dir, building, buildings):
    """Generate HTML page for a single building."""
    summaries = load_summaries(results_dir, building)

    # Build rewards table data
    rewards = {}
    for s in summaries:
        rewards[(s['policy'], s['weather'])] = s['total_reward']

    # Calculate min/max for coloring
    if rewards:
        min_r = min(rewards.values())
        max_r = max(rewards.values())
        range_r = max_r - min_r if max_r != min_r else 1
    else:
        min_r = max_r = range_r = 0

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{building} - Smart Buildings Results</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        .nav {{ margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 8px; }}
        .nav a {{
            padding: 8px 16px; background: #4a90d9; color: white;
            text-decoration: none; border-radius: 4px; font-size: 14px;
        }}
        .nav a:hover {{ background: #357abd; }}
        .nav a.active {{ background: #2d6da3; }}
        .section {{
            background: white; padding: 20px; margin-bottom: 20px;
            border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
        th {{ background: #4a90d9; color: white; }}
        .policy-header {{ background: #5ba55b !important; color: white; font-weight: bold; }}
        .reward-high {{ background: #c8e6c9; }}
        .reward-mid {{ background: #fff9c4; }}
        .reward-low {{ background: #ffcdd2; }}
        .pending {{ color: #999; }}
        .gif-grid {{
            display: grid;
            grid-template-columns: auto repeat(6, 1fr);
            gap: 8px;
            align-items: start;
        }}
        .gif-grid img {{
            width: 100%; max-width: 200px; border-radius: 4px;
            transition: transform 0.2s;
        }}
        .gif-grid img:hover {{ transform: scale(1.02); }}
        .gif-cell {{ text-align: center; }}
        .header-cell {{ font-weight: bold; padding: 8px; font-size: 12px; }}
        .weather-header {{ background: #4a90d9; color: white; border-radius: 4px; }}
        .policy-cell {{
            background: #5ba55b; color: white; border-radius: 4px;
            writing-mode: vertical-rl; text-orientation: mixed;
            padding: 15px 8px;
        }}
        @media (max-width: 900px) {{
            .gif-grid {{ grid-template-columns: auto repeat(3, 1fr); }}
        }}
    </style>
</head>
<body>
    <h1>Smart Buildings Simulation Results</h1>

    <div class="nav">
'''

    # Navigation links
    for b in buildings:
        active = "active" if b == building else ""
        html += f'        <a href="{b}.html" class="{active}">{b}</a>\n'

    html += '''    </div>

    <div class="section">
        <h2>Rewards Heatmap</h2>
        <table>
            <tr><th></th>'''

    for w in WEATHERS:
        html += f'<th>{WEATHER_NAMES[w]}</th>'
    html += '</tr>\n'

    for policy in POLICIES:
        html += f'            <tr><td class="policy-header">{POLICY_NAMES[policy]}</td>'
        for weather in WEATHERS:
            key = (policy, weather)
            if key in rewards:
                r = rewards[key]
                pct = (r - min_r) / range_r if range_r else 0
                if pct > 0.66:
                    cls = "reward-high"
                elif pct > 0.33:
                    cls = "reward-mid"
                else:
                    cls = "reward-low"
                html += f'<td class="{cls}">{r:.1f}</td>'
            else:
                html += '<td class="pending">-</td>'
        html += '</tr>\n'

    html += '''        </table>
    </div>

    <div class="section">
        <h2>Simulation Results</h2>
        <p><em>Overview snapshots for each weather/policy combination</em></p>
        <div class="gif-grid">
            <div></div>
'''

    for w in WEATHERS:
        html += f'            <div class="header-cell weather-header">{WEATHER_NAMES[w]}</div>\n'

    for policy in POLICIES:
        html += f'            <div class="header-cell policy-cell">{POLICY_NAMES[policy]}</div>\n'
        for weather in WEATHERS:
            png_path = results_dir / building / weather / policy / "overview.png"

            if png_path.exists():
                png_rel = f"{building}/{weather}/{policy}/overview.png"
                html += f'            <div class="gif-cell"><img src="{png_rel}" loading="lazy"></div>\n'
            else:
                html += '            <div class="gif-cell pending">pending</div>\n'

    html += '''        </div>
    </div>
</body>
</html>'''

    # Write the HTML file
    output_path = output_dir / f"{building}.html"
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"  Generated {output_path}")


def copy_assets(results_dir, output_dir, building):
    """Copy images for a building (no GIFs)."""
    for weather in WEATHERS:
        for policy in POLICIES:
            src_dir = results_dir / building / weather / policy
            dst_dir = output_dir / building / weather / policy

            if not src_dir.exists():
                continue

            dst_dir.mkdir(parents=True, exist_ok=True)

            # Copy only static images (no GIFs)
            for filename in ["overview.png", "rewards.png", "ahu.png"]:
                src = src_dir / filename
                if src.exists():
                    shutil.copy2(src, dst_dir / filename)


def generate_index(output_dir, buildings):
    """Generate index.html that redirects to first building."""
    if not buildings:
        html = "<html><body><h1>No results yet</h1></body></html>"
    else:
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url={buildings[0]}.html">
</head>
<body>
    <p>Redirecting to <a href="{buildings[0]}.html">{buildings[0]}</a>...</p>
</body>
</html>'''

    with open(output_dir / "index.html", 'w') as f:
        f.write(html)
    print(f"  Generated {output_dir / 'index.html'}")


def main():
    parser = ArgumentParser(description="Generate static site from simulation results")
    parser.add_argument("results_dir", nargs="?", default="distributed_results",
                        help="Directory containing results (default: distributed_results)")
    parser.add_argument("output_dir", nargs="?", default="docs",
                        help="Output directory for static site (default: docs)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print(f"Generating static site...")
    print(f"  Results: {results_dir}")
    print(f"  Output:  {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get buildings
    buildings = get_buildings(results_dir)
    print(f"  Found {len(buildings)} buildings: {', '.join(buildings)}")

    if not buildings:
        print("No results found!")
        return

    # Generate pages and copy assets for each building
    for building in buildings:
        print(f"\nProcessing {building}...")
        generate_building_page(results_dir, output_dir, building, buildings)
        copy_assets(results_dir, output_dir, building)

    # Generate index
    print(f"\nGenerating index...")
    generate_index(output_dir, buildings)

    print(f"\n✓ Site generated in {output_dir}/")
    print(f"\nTo view locally:")
    print(f"  cd {output_dir} && python -m http.server 8000")
    print(f"  Then open http://localhost:8000")
    print(f"\nTo deploy to GitHub Pages:")
    print(f"  1. git add {output_dir}")
    print(f"  2. git commit -m 'Update results site'")
    print(f"  3. git push")
    print(f"  4. Go to repo Settings > Pages > Source: Deploy from branch, folder: /docs")


if __name__ == "__main__":
    main()
