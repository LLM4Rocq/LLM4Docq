# generate_retro_pixel_fixed.py
"""
Generate retro '90s video game'-styled PNGs for:
  - Completion state (progress bars)
  - Leaderboard

Requires a pixelated TTF font in the working directory.

Usage:
  python generate_retro_pixel_fixed.py --input data.json --font export/lilliput_steps.otf --output-dir export
"""

import os
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

def register_font(font_path):
    """Register and return matplotlib FontProperties for the given TTF."""
    if not os.path.isfile(font_path):
        raise FileNotFoundError(f"Font file not found: {font_path}")
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    # Set global rcParams to use this font family
    font_name = prop.get_name()
    plt.rcParams['font.family'] = font_name
    return prop

def generate_progress(sections, output_dir, font_path):
    """
    For each section in `sections` (a dict mapping section_name -> {module_name: score, ...}),
    generate a separate PNG file showing that section's progress bars.

    Args:
        sections (dict[str, dict[str, int]]):
            Outer dict keys are section names; values are dicts mapping module/label -> percentage (0–100).
        output_dir (str or Path):
            Directory where individual PNGs will be saved. Will be created if it doesn't exist.
        font_path (str or Path):
            Path to a .ttf/.otf font file for rendering labels and titles.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    font_prop = register_font(font_path)
    max_rows = 5  # number of bars per column

    for section_name, modules in sections.items():
        # Determine how many columns are needed for this section
        n_items = len(modules)
        sec_cols = math.ceil(n_items / max_rows) if n_items > 0 else 1

        # Compute figure dimensions:
        #   - Each column is width ≈ 4 inches
        #   - Row height: max_rows * 0.5 (bar height & spacing) + 1 inch padding for title
        row_height = max_rows * 0.5 + 1  # e.g., 5*0.5 + 1 = 3.5 inches
        fig_width = sec_cols * 4
        fig_height = row_height

        fig, axes = plt.subplots(
            nrows=1,
            ncols=sec_cols,
            figsize=(fig_width, fig_height),
            facecolor='black',
            gridspec_kw={'wspace': 1.0},
            constrained_layout=True
        )

        # Ensure axes is a 1D list
        if sec_cols == 1:
            axes = [axes]
        else:
            axes = list(axes)

        # Plot the bars for this section
        items = list(modules.items())  # list of (label, score)
        # Pad items so that total = sec_cols * max_rows
        total_slots = sec_cols * max_rows
        padded = items.copy()
        padded += [("", 0)] * (total_slots - len(items))

        for col_idx in range(sec_cols):
            ax = axes[col_idx]
            ax.set_facecolor('black')

            # Slice out the rows for this column
            block = padded[col_idx * max_rows : (col_idx + 1) * max_rows]
            labels, scores = zip(*block)
            fracs = [s / 100.0 for s in scores]
            y = list(range(max_rows))

            ax.barh(y, fracs, color='#00FF00', edgecolor='white', height=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, color='white', fontproperties=font_prop)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.invert_yaxis()

            # Annotate each bar with its percentage
            for i, score in enumerate(scores):
                if labels[i]:
                    ax.text(fracs[i] + 0.02, i, f"{score}%", va='center',
                            color='white', fontproperties=font_prop, fontsize=10)

        # Add the section title centered above all columns
        # Compute leftmost and rightmost axes positions
        pos0 = axes[0].get_position()
        posn = axes[-1].get_position()
        center_x = (pos0.x0 + posn.x1) / 2
        title_y = pos0.y1 + 0.06
        fig.text(
            center_x, title_y,
            section_name.upper(),
            ha='center',
            va='bottom',
            color='white',
            fontproperties=font_prop,
            fontsize=20
        )
        # Save one PNG per section, sanitize filename
        safe_name = section_name.replace(".", "_").replace(" ", "_")
        output_path = output_dir / f"{safe_name}.png"
        fig.savefig(output_path, dpi=150, facecolor='black')
        plt.close(fig)

def generate_leaderboard(data, output_path, font_path):
    font_prop = register_font(font_path)
    lb = data.get("leaderboard", {})
    sorted_lb = sorted(lb.items(), key=lambda x: -x[1])
    names = [n for n, _ in sorted_lb]
    scores = [s for _, s in sorted_lb]
    max_score = max(scores) if scores else 1
    fracs = [s / max_score for s in scores]

    fig, ax = plt.subplots(figsize=(6, len(names)*0.6 + 1), facecolor='black')
    ax.set_facecolor('black')
    y = list(range(len(names)))

    ax.barh(y, fracs, color='#00FF00', edgecolor='white', height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color='white', fontproperties=font_prop)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("LEADERBOARD", color='white', pad=20, fontproperties=font_prop)

    for i, score in enumerate(scores):
        ax.text(fracs[i] + 0.02, i, str(score), va='center', color='white', fontproperties=font_prop)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor='black')
    plt.close(fig)