#!/usr/bin/env python3
"""Procedural Floorplan Generator - generates diverse building floorplans."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import random
from dataclasses import dataclass
from pathlib import Path
import argparse

@dataclass
class Room:
    x: int
    y: int
    width: int
    height: int
    room_type: str
    name: str = ""

ROOM_COLORS = {
    "office": "#E3F2FD",
    "conference": "#FFF3E0",
    "lobby": "#E8F5E9",
    "restroom": "#FCE4EC",
    "kitchen": "#FFF8E1",
    "server": "#E0E0E0",
    "storage": "#EFEBE9",
    "elevator": "#B0BEC5",
    "stairs": "#CFD8DC",
    "open_office": "#E1F5FE",
    "lab": "#E0F7FA",
}

STYLES = {
    "colorful": {"bg": "#eceff1", "wall": "#37474f", "corridor": "#b0bec5"},
    "blueprint": {"bg": "#1a237e", "wall": "#e3f2fd", "corridor": "#3949ab"},
    "modern": {"bg": "#263238", "wall": "#80cbc4", "corridor": "#455a64"},
}

# ============ GENERATORS ============

def generate_office_floor(width=60, height=40, seed=None):
    """Standard office with central corridor."""
    if seed: random.seed(seed)
    rooms = []
    corridor_y = height // 2 - 2
    corridor_h = 4

    # Lobby
    rooms.append(Room(2, corridor_y - 6, 10, 16, "lobby", "Lobby"))

    # Core (elevator, stairs, restrooms)
    cx = width // 2 - 4
    rooms.append(Room(cx, corridor_y - 5, 4, 5, "elevator", "Elev"))
    rooms.append(Room(cx + 4, corridor_y - 5, 4, 5, "stairs", "Stairs"))
    rooms.append(Room(cx, corridor_y + corridor_h, 4, 5, "restroom", "WC"))
    rooms.append(Room(cx + 4, corridor_y + corridor_h, 4, 5, "restroom", "WC"))

    # Top offices
    x, num = 14, 101
    while x < width - 8:
        if cx - 2 <= x <= cx + 10:
            x = cx + 11
            continue
        w = random.choice([6, 7, 8])
        if x + w > width - 2:
            break
        rooms.append(Room(x, 2, w, corridor_y - 3, random.choice(["office"]*3 + ["conference"]), str(num)))
        x += w + 1
        num += 1

    # Bottom offices
    x, num = 14, 201
    while x < width - 8:
        if cx - 2 <= x <= cx + 10:
            x = cx + 11
            continue
        w = random.choice([6, 7, 8])
        if x + w > width - 2:
            break
        rooms.append(Room(x, corridor_y + corridor_h + 1, w, height - corridor_y - corridor_h - 3,
                         random.choice(["office"]*2 + ["conference", "kitchen"]), str(num)))
        x += w + 1
        num += 1

    return rooms, width, height, [(2, corridor_y, width - 4, corridor_h)]


def generate_open_plan(width=60, height=40, seed=None):
    """Open plan office with perimeter offices."""
    if seed: random.seed(seed)
    rooms = []
    margin = 8

    # Central open area
    rooms.append(Room(margin, margin, width - 2*margin, height - 2*margin, "open_office", "Open Office"))

    # Perimeter offices
    for i in range(2, width - 6, 7):
        rooms.append(Room(i, 1, 6, margin - 2, random.choice(["office", "conference"])))
        rooms.append(Room(i, height - margin + 1, 6, margin - 2, random.choice(["office", "conference"])))

    # Core
    cx, cy = width // 2 - 6, height // 2 - 4
    rooms.append(Room(cx, cy, 4, 4, "elevator"))
    rooms.append(Room(cx + 4, cy, 4, 4, "stairs"))
    rooms.append(Room(cx + 8, cy, 4, 4, "restroom"))
    rooms.append(Room(cx, cy + 4, 6, 4, "kitchen"))
    rooms.append(Room(cx + 6, cy + 4, 6, 4, "server"))

    return rooms, width, height, []


def generate_lab_floor(width=70, height=50, seed=None):
    """Research lab with L-shaped corridor."""
    if seed: random.seed(seed)
    rooms = []
    corridors = [(2, height // 2 - 2, width - 4, 4), (width // 3, 2, 4, height - 4)]

    # Labs on left
    for i, y in enumerate(range(2, height // 2 - 4, 10)):
        rooms.append(Room(2, y, width // 3 - 4, 9, "lab", f"Lab {i+1}"))
    for i, y in enumerate(range(height // 2 + 4, height - 10, 10)):
        rooms.append(Room(2, y, width // 3 - 4, 9, "lab", f"Lab {i+3}"))

    # Offices on right
    x = width // 3 + 6
    while x < width - 10:
        w = random.choice([7, 8, 9])
        rooms.append(Room(x, 2, w, height // 2 - 5, random.choice(["office", "conference"])))
        rooms.append(Room(x, height // 2 + 3, w, height // 2 - 5, random.choice(["office", "storage"])))
        x += w + 1

    # Core
    rooms.append(Room(width - 10, height // 2 - 6, 8, 5, "elevator"))
    rooms.append(Room(width - 10, height // 2 + 3, 8, 5, "stairs"))

    return rooms, width, height, corridors


def generate_random_floor(width=60, height=40, seed=None):
    """Random BSP-generated layout."""
    if seed: random.seed(seed)
    rooms = []

    def split(x, y, w, h, depth=0):
        if depth > 3 or w < 10 or h < 10:
            rooms.append(Room(x + 1, y + 1, w - 2, h - 2, random.choice(list(ROOM_COLORS.keys()))))
            return
        if random.random() < 0.25 and depth > 1:
            rooms.append(Room(x + 1, y + 1, w - 2, h - 2, random.choice(list(ROOM_COLORS.keys()))))
            return
        if w > h:
            s = random.randint(w // 3, 2 * w // 3)
            split(x, y, s, h, depth + 1)
            split(x + s, y, w - s, h, depth + 1)
        else:
            s = random.randint(h // 3, 2 * h // 3)
            split(x, y, w, s, depth + 1)
            split(x, y + s, w, h - s, depth + 1)

    split(2, 2, width - 4, height - 4)
    rooms.append(Room(width // 2 - 3, height // 2 - 3, 3, 3, "elevator"))
    rooms.append(Room(width // 2, height // 2 - 3, 3, 3, "stairs"))

    return rooms, width, height, []


# ============ RENDERER ============

def render_floor(rooms, width, height, corridors, title="Floor", style="colorful"):
    """Render a single floor."""
    s = STYLES.get(style, STYLES["colorful"])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor(s["bg"])

    # Corridors
    for cx, cy, cw, ch in corridors:
        ax.add_patch(Rectangle((cx, cy), cw, ch, facecolor=s["corridor"], edgecolor=s["wall"], lw=0.5))

    # Rooms
    for room in rooms:
        color = ROOM_COLORS.get(room.room_type, "#fff") if style == "colorful" else s["bg"]
        ax.add_patch(FancyBboxPatch((room.x, room.y), room.width, room.height,
            boxstyle="round,pad=0.02", facecolor=color, edgecolor=s["wall"], lw=1.5))
        if room.width > 4 and room.height > 3:
            txt_color = "#333" if style == "colorful" else s["wall"]
            ax.text(room.x + room.width/2, room.y + room.height/2,
                   room.name or room.room_type[:6], ha='center', va='center',
                   fontsize=7, color=txt_color, weight='bold')

    # Outline
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor=s["wall"], lw=3))
    ax.set_xlim(-2, width + 2)
    ax.set_ylim(-2, height + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', color=s["wall"], pad=10)
    plt.tight_layout()
    return fig


def render_multifloor(floors_data, title="Building", style="colorful"):
    """Render multiple floors in a grid."""
    n = len(floors_data)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    s = STYLES.get(style, STYLES["colorful"])

    fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 8 * rows))
    fig.patch.set_facecolor(s["bg"])
    axes = [axes] if n == 1 else axes.flatten()

    for i, (rooms, width, height, corridors, floor_num) in enumerate(floors_data):
        ax = axes[i]
        ax.set_facecolor(s["bg"])

        for cx, cy, cw, ch in corridors:
            ax.add_patch(Rectangle((cx, cy), cw, ch, facecolor=s["corridor"], edgecolor=s["wall"], lw=0.5))

        for room in rooms:
            color = ROOM_COLORS.get(room.room_type, "#fff") if style == "colorful" else s["bg"]
            ax.add_patch(FancyBboxPatch((room.x, room.y), room.width, room.height,
                boxstyle="round,pad=0.02", facecolor=color, edgecolor=s["wall"], lw=1.5))
            if room.width > 4 and room.height > 3:
                txt_color = "#333" if style == "colorful" else s["wall"]
                ax.text(room.x + room.width/2, room.y + room.height/2,
                       room.name or room.room_type[:6], ha='center', va='center',
                       fontsize=7, color=txt_color, weight='bold')

        ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor=s["wall"], lw=3))
        ax.set_xlim(-2, width + 2)
        ax.set_ylim(-2, height + 2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Floor {floor_num}", fontsize=12, fontweight='bold', color=s["wall"])

    for i in range(n, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold', color=s["wall"], y=0.98)
    plt.tight_layout()
    return fig


def to_numpy(rooms, width, height, scale=4):
    """Convert to simulation array: 0=interior, 1=wall, 2=exterior."""
    arr = np.full((height * scale, width * scale), 2, dtype=np.uint8)
    arr[scale:-scale, scale:-scale] = 0

    for room in rooms:
        x1, y1 = room.x * scale, room.y * scale
        x2, y2 = (room.x + room.width) * scale, (room.y + room.height) * scale
        arr[y1:y1+1, x1:x2] = 1
        arr[y2-1:y2, x1:x2] = 1
        arr[y1:y2, x1:x1+1] = 1
        arr[y1:y2, x2-1:x2] = 1
        # Doorway
        cx = (room.x + room.width // 2) * scale
        cy = (room.y + room.height) * scale - 1
        arr[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = 0

    return arr


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description="Generate procedural building floorplans")
    parser.add_argument("-n", "--num", type=int, default=6, help="Number of floorplans to generate")
    parser.add_argument("-o", "--output", type=str, default="floorplans", help="Output directory")
    parser.add_argument("-s", "--style", choices=["colorful", "blueprint", "modern"], default="colorful")
    parser.add_argument("--multi", type=int, default=0, help="Generate multi-floor building with N floors")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--numpy", action="store_true", help="Also export numpy arrays for simulation")
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(exist_ok=True)

    generators = [generate_office_floor, generate_open_plan, generate_lab_floor, generate_random_floor]
    gen_names = ["office", "openplan", "lab", "random"]
    styles = ["colorful", "blueprint", "modern"]

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.multi > 0:
        # Generate multi-floor building
        floors_data = []
        for i in range(args.multi):
            gen = generators[i % len(generators)]
            seed = np.random.randint(0, 10000)
            rooms, w, h, corr = gen(seed=seed)
            floors_data.append((rooms, w, h, corr, i + 1))

        fig = render_multifloor(floors_data, f"{args.multi}-Floor Building", args.style)
        outpath = outdir / f"multifloor_{args.multi}.png"
        fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {outpath}")

        if args.numpy:
            for rooms, w, h, _, floor_num in floors_data:
                arr = to_numpy(rooms, w, h)
                np.save(outdir / f"floor_{floor_num}.npy", arr)
                print(f"Saved: floor_{floor_num}.npy")
    else:
        # Generate individual floorplans
        for i in range(args.num):
            seed = np.random.randint(0, 10000)
            gen_idx = i % len(generators)
            gen = generators[gen_idx]
            style = styles[i % len(styles)] if args.style == "colorful" else args.style

            rooms, w, h, corr = gen(seed=seed)
            title = f"{gen_names[gen_idx].title()} (seed={seed})"

            fig = render_floor(rooms, w, h, corr, title, style)
            outpath = outdir / f"floorplan_{i+1:02d}_{gen_names[gen_idx]}.png"
            fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Saved: {outpath}")

            if args.numpy:
                arr = to_numpy(rooms, w, h)
                np.save(outdir / f"floorplan_{i+1:02d}.npy", arr)

    print(f"\nDone! Generated {args.multi if args.multi else args.num} floorplan(s) in {outdir}/")


if __name__ == "__main__":
    main()
