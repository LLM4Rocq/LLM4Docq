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

def generate_progress(data, output_path, font_path):
    font_prop = register_font(font_path)
    sections = data.get("global", {})
    n_sections = len(sections)
    # Determine columns needed per section

    max_rows = 5
    cols_per_sec = [math.ceil(len(mods) / max_rows) for mods in sections.values()]
    max_cols = max(cols_per_sec) if cols_per_sec else 1

    # Compute figure size
    row_height = max_rows * 0.5 + 1  # 10 rows at 0.5 each + title pad
    fig_width = max_cols * 4
    fig_height = n_sections * row_height

    fig, axes = plt.subplots(nrows=n_sections, ncols=max_cols, gridspec_kw={'hspace':1.0},
                             figsize=(fig_width, fig_height), 
                             facecolor='black')

    # Ensure axes is 2D array
    if n_sections == 1 and max_cols == 1:
        axes = [[axes]]
    elif n_sections == 1:
        axes = [axes]
    elif max_cols == 1:
        axes = [[ax] for ax in axes]

    # Plot bars and prepare for titles
    for sec_idx, ((section, mods), sec_cols) in enumerate(zip(sections.items(), cols_per_sec)):
        for col_idx in range(max_cols):
            ax = axes[sec_idx][col_idx]
            ax.set_facecolor('black')
            if col_idx < sec_cols:
                # Slice items for this column
                items = list(mods.items())[col_idx*max_rows:(col_idx+1)*max_rows]
                # Pad to max_rows
                while len(items) < max_rows:
                    items.append(("", 0))
                labels, scores = zip(*items)
                fracs = [s / 100.0 for s in scores]
                y = list(range(max_rows))

                ax.barh(y, fracs, color='#00FF00', edgecolor='white', height=0.6)
                ax.set_yticks(y)
                ax.set_yticklabels(labels, color='white', fontproperties=font_prop)
                ax.set_xlim(0, 1)
                ax.set_xticks([])
                ax.invert_yaxis()
                # Annotate scores
                for i, score in enumerate(scores):
                    if labels[i]:
                        ax.text(fracs[i] + 0.02, i, f"{score}%", va='center', color='white', fontproperties=font_prop)
            else:
                # Empty subplot
                ax.axis('off')

    # Centered section titles
    for sec_idx, (section, _) in enumerate(sections.items()):
        row_axes = axes[sec_idx]
        pos0 = row_axes[0].get_position()
        posn = row_axes[-1].get_position()
        center = (pos0.x0 + posn.x1) / 2
        # Place title above the plots
        title_y = pos0.y1 + 0.06
        fig.text(center, title_y, section.upper(), ha='center', va='bottom',
                 color='white', fontproperties=font_prop, fontsize=20)

    plt.tight_layout()
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