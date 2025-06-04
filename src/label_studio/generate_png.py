# generate_retro_pixel_fixed.py
"""
Generate retro '90s video game'-styled PNGs for:
  - Completion state (progress bars)
  - Leaderboard

Requires a pixelated TTF font in the working directory.

Usage:
  python generate_retro_pixel_fixed.py --input data.json --font export/lilliput_steps.otf --output-dir export
"""

import math
from pathlib import Path

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
import matplotlib.pyplot as plt

def register_font(font_path, size):
    """
    Return a FontProperties object for the given font file and size.
    """
    prop = FontProperties(fname=font_path, size=size)
    return prop

def get_text_width_inch(text, font_prop):
    """
    Measure the width of `text` in inches using a TextPath and FontProperties.
    """
    tp = TextPath((0, 0), text, prop=font_prop)
    bbox = tp.get_extents()
    width_pt = bbox.width   # width in points
    return width_pt / 72

def generate_progress(sections, output_dir, font_path):
    """
    For each section, generate a PNG with exactly 3 horizontal‐bar columns,
    balancing the modules as evenly as possible across those 3 columns, and
    centering the title via fig.suptitle(x=0.5).

    Args:
        sections (dict[str, dict[str, int]]):
            Mapping from section name to {module_name: percentage (0–100)}.
        output_dir (str or Path):
            Directory where output PNGs will be saved.
        font_path (str or Path):
            Path to a .ttf or .otf font file for rendering text.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sec_cols = 3  # always use 3 columns

    for section_name, modules in sections.items():
        # 1) Empty‐section case
        if not modules:
            fig = plt.figure(figsize=(4, 2), facecolor='black')
            title_fp = register_font(font_path, size=20)
            fig.suptitle(
                section_name.upper(),
                x=0.5, y=0.5,
                ha='center', va='center',
                color='white',
                fontproperties=title_fp,
                fontsize=20
            )
            safe = section_name.replace(".", "_").replace(" ", "_")
            out_path = output_dir / f"{safe}.png"
            fig.savefig(out_path, dpi=150, facecolor='black', bbox_inches='tight')
            plt.close(fig)
            print(f"Saved empty section title: {out_path}")
            continue

        # 2) Determine rows_per_col to spread modules evenly across 3 columns
        items = list(modules.items())
        n_items = len(items)
        rows_per_col = math.ceil(n_items / sec_cols)

        # 3) Measure text widths (for left/right margins)
        label_fp = register_font(font_path, size=8)
        annot_fp = register_font(font_path, size=10)

        # a) Longest module name (in inches)
        max_label_w = 0
        for lbl in modules.keys():
            w = get_text_width_inch(lbl, label_fp)
            max_label_w = max(max_label_w, w)
        left_margin_in = max_label_w + 0.1  # 0.1" padding

        # b) “100%” width (in inches)
        annot_example = "100%"
        annot_w = get_text_width_inch(annot_example, annot_fp)
        right_margin_in = annot_w + 0.1

        # 4) Compute figure‐height and ‐width (in inches)
        #    Height = rows_per_col * 0.5 + 1 (0.5" per row + 1" title padding)
        row_h = rows_per_col * 0.5 + 1
        fig_h = max(row_h, 2)  # at least 2"

        #    Width: ensure at least 4" per column OR enough to fit margins + bars
        #    (so bars get ~2" each if many modules), but at least 4" total.
        fig_w = max(
            sec_cols * 4,                              # 4" per column baseline
            left_margin_in + (sec_cols * 2) + right_margin_in,
            4                                           # at least 4" total
        )

        # 5) Compute wspace so there’s ~0.2" gap between columns
        total_axes_w = fig_w - left_margin_in - right_margin_in
        axes_w = total_axes_w / sec_cols
        desired_space_in = 0.2
        wspace = (desired_space_in / axes_w) + 1.3

        # 6) Create subplots (3 columns, no constrained_layout)
        fig, axes = plt.subplots(
            nrows=1,
            ncols=sec_cols,
            figsize=(fig_w, fig_h),
            facecolor='black',
            constrained_layout=False
        )
        # Normalize axes list
        if sec_cols == 1:
            axes = [axes]
        else:
            axes = list(axes)

        # 7) Pad items so that total slots = 3 * rows_per_col
        total_slots = sec_cols * rows_per_col
        padded = items + [("", 0)] * (total_slots - n_items)

        # 8) Plot each of the 3 columns
        for col_idx in range(sec_cols):
            ax = axes[col_idx]
            ax.set_facecolor('black')

            block = padded[col_idx * rows_per_col : (col_idx + 1) * rows_per_col]
            labels, scores = zip(*block)
            fracs = [s / 100.0 for s in scores]
            y = list(range(rows_per_col))

            ax.barh(y, fracs, color='#00FF00', edgecolor='white', height=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, color='white',
                               fontproperties=label_fp, fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.invert_yaxis()

            for i, sc in enumerate(scores):
                if labels[i]:
                    frac = fracs[i]
                    if frac >= 0.8:
                        xtext = frac - 0.02
                        ha = 'right'
                    else:
                        xtext = frac + 0.02
                        ha = 'left'
                    ax.text(
                        xtext, i, f"{sc}%",
                        va='center', ha=ha,
                        color='white',
                        fontproperties=annot_fp,
                        fontsize=10
                    )

        # 9) Convert margins from inches → figure fractions
        left_frac  = left_margin_in / fig_w
        right_frac = 1 - (right_margin_in / fig_w)

        fig.subplots_adjust(
            left=left_frac,
            right=right_frac,
            top=0.85,      # leave room for suptitle
            bottom=0.05,
            wspace=wspace
        )

        # 10) Centered section title
        title_fp = register_font(font_path, size=20)
        fig.suptitle(
            section_name.upper(),
            x=0.5, y=0.95,
            ha='center', va='top',
            color='white',
            fontproperties=title_fp,
            fontsize=20
        )

        # 11) Save and close
        safe = section_name.replace(".", "_").replace(" ", "_")
        out_path = output_dir / f"{safe}.png"
        fig.savefig(out_path, dpi=150, facecolor='black', bbox_inches='tight')
        plt.close(fig)

def generate_leaderboard(data, output_path, font_path, top_k=8):
    """
    Generate a '90s video game'–style leaderboard PNG with exactly `top_k` rows.
    If there are fewer than top_k players, the remaining rows are left blank.
    The title is centered at the top.

    Args:
        data (dict[str,int]): Mapping from player_name -> score.
        output_path (str or Path): Where to write the PNG.
        font_path (str or Path): Path to a .ttf/.otf font file.
        top_k (int): How many total rows (including empty) to show.
    """
    # 1) Sort descending by score and take up to top_k entries
    sorted_lb = sorted(data.items(), key=lambda x: x[1], reverse=True)
    actual = sorted_lb[:top_k]  # may be fewer than top_k

    # 2) Separate names & scores
    names  = [n for n, _ in actual]
    scores = [s for _, s in actual]

    # 3) Pad with empty rows if needed
    n_actual = len(names)
    if n_actual < top_k:
        names += [""] * (top_k - n_actual)
        scores += [0] * (top_k - n_actual)

    # 4) If every name is empty, just write a title‐only PNG
    if all(name == "" for name in names):
        fig = plt.figure(figsize=(4, 2), facecolor="black")
        title_fp = register_font(font_path, size=20)
        fig.suptitle(
            "LEADERBOARD",
            x=0.5, y=0.5,
            ha="center", va="center",
            color="white",
            fontproperties=title_fp,
            fontsize=20
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, facecolor="black", bbox_inches="tight")
        plt.close(fig)
        return

    # 5) Compute fractions (avoid division by zero)
    max_score = max(scores) if max(scores) > 0 else 1
    fracs = [s / max_score for s in scores]

    # 6) Prepare FontProperties
    label_fp  = register_font(font_path, size=10)
    annot_fp  = register_font(font_path, size=10)
    title_fp  = register_font(font_path, size=20)

    # 7a) Compute left margin: measure only non‐empty, string names
    valid_names = [n for n in names if isinstance(n, str) and n != ""]
    max_name_w = max((get_text_width_inch(n, label_fp) for n in valid_names), default=0.0)
    left_margin_in = max_name_w + 0.1

    # 7b) Compute right margin: width of the largest possible score (as a string)
    largest_str = str(max_score)
    annot_w = get_text_width_inch(largest_str, annot_fp)
    right_margin_in = annot_w + 0.1

    # 8) Compute figure size (in inches)
    row_h = top_k * 0.6 + 1        # 0.6" per row + 1" for title padding
    fig_h = max(row_h, 2)          # at least 2" tall
    min_axes_w = 4
    fig_w = max(left_margin_in + min_axes_w + right_margin_in, 6)

    # 9) Convert margins to fraction of figure width
    left_frac  = left_margin_in / fig_w
    right_frac = 1 - (right_margin_in / fig_w)

    # 10) Create the figure & axis
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="black")
    ax.set_facecolor("black")

    # 11) Draw bars for all top_k rows
    y = list(range(top_k))
    ax.barh(y, fracs, color="#00FF00", edgecolor="white", height=0.6)

    # 12) Set y‐tick labels (some may be "")
    ax.set_yticks(y)
    ax.set_yticklabels(names, color="white", fontproperties=label_fp, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])

    # 13) Annotate each bar’s score (skip empty rows)
    for i, (frac, sc, nm) in enumerate(zip(fracs, scores, names)):
        if isinstance(nm, str) and nm and sc > 0:
            sc_str = str(sc)
            if frac >= 0.8:
                xtext = frac - 0.02
                ha = "right"
            else:
                xtext = frac + 0.02
                ha = "left"
            ax.text(
                xtext, i, sc_str,
                va="center", ha=ha,
                color="white",
                fontproperties=annot_fp,
                fontsize=10
            )

    # 14) Centered title via fig.suptitle(x=0.5)
    fig.suptitle(
        "LEADERBOARD",
        x=0.5, y=0.95,
        ha="center", va="top",
        color="white",
        fontproperties=title_fp,
        fontsize=20
    )

    # 15) Adjust subplot so nothing is clipped
    fig.subplots_adjust(
        left=left_frac,
        right=right_frac,
        top=0.90,    # a little room for the title above
        bottom=0.05
    )

    # 16) Save the figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sections = {
        "global": {
            "mathcomp.algebra.mxpoly": 80,
            "mathcomp.algebra.ssrint": 15,
            "mathcomp.algebra.vector": 40,
            "mathcomp.algebra.finalg": 100,
            "some.super.long.module.name": 60,
            "mathcomp.algebra.mxpoly1": 80,
            "mathcomp.algebra.ssrint1": 15,
            "mathcomp.algebra.vector1": 40,
            "mathcomp.algebra.finalg1": 100,
            "some.super.long.module.name1": 60,
            "mathcomp.algebra.mxpoly2": 80,
            "mathcomp.algebra.ssrint2": 15,
            "mathcomp.algebra.vector2": 40,
            "mathcomp.algebra.finalg2": 100,
            "some.super.long.module.name2": 60
        },
        "test": {
            "mathcomp.algebra.mxpoly": 80,
            "mathcomp.algebra.ssrint": 15,
            "mathcomp.algebra.vector": 40,
            "mathcomp.algebra.finalg": 100,
            "some.super.long.module.name": 60,
            "mathcomp.algebra.mxpoly1": 80,
            "mathcomp.algebra.ssrint1": 15,
            "mathcomp.algebra.vector1": 40,
            "mathcomp.algebra.finalg1": 100,
            "some.super.long.module.name1": 60
        },
    }

    leaderboard = {
        "name": 100,
        "veryveryveryveryveryverylongname": 142,
        "test": 0,
        "name2": 10,
        "algebra_finalg1": 250,
        "blopablopinria": 40,

    }
    # 3) Pick a font that you know exists on your system:
    font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    # 4) Call the function:
    generate_progress(
        sections=sections,
        output_dir="export",
        font_path=font_file
    )

    generate_leaderboard(leaderboard, output_path="export/leaderboard_test.png", font_path=font_file)
