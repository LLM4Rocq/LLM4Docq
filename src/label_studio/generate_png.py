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
    For each section, generate a PNG with horizontal bars, and place the section title
    centered at the top of the figure via fig.suptitle(x=0.5).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_rows = 5

    for section_name, modules in sections.items():
        # 1) Handle “empty” section (no modules)
        if not modules:
            fig = plt.figure(figsize=(4, 2), facecolor='black')
            title_prop = register_font(font_path, size=20)
            fig.suptitle(
                section_name.upper(),
                x=0.5, y=0.5,
                ha='center', va='center',
                color='white',
                fontproperties=title_prop,
                fontsize=20
            )
            safe_name = section_name.replace(".", "_").replace(" ", "_")
            out_path = output_dir / f"{safe_name}.png"
            fig.savefig(out_path, dpi=150, facecolor='black', bbox_inches='tight')
            plt.close(fig)
            print(f"Saved empty section title: {out_path}")
            continue

        # 2) Determine how many columns we need
        n_items = len(modules)
        sec_cols = math.ceil(n_items / max_rows)

        # 3) Measure text widths (to set margins)
        label_fp = register_font(font_path, size=8)
        annot_fp = register_font(font_path, size=10)

        # a) Longest module name
        max_label_w = 0
        for lbl in modules.keys():
            w = get_text_width_inch(lbl, label_fp)
            max_label_w = max(max_label_w, w)
        left_margin_in = max_label_w + 0.1  # add 0.1" padding on left

        # b) “100%” width
        annot_example = "100%"
        annot_w = get_text_width_inch(annot_example, annot_fp)
        right_margin_in = annot_w + 0.1

        # 4) Compute overall figure size (in inches)
        row_h = max_rows * 0.5 + 1           # e.g. 5*0.5 + 1 = 3.5"
        fig_h = max(row_h, 2)               # at least 2" tall

        # Guarantee at least 4" per column, or enough to fit margins+bars
        fig_w = max(
            sec_cols * 4,                             # 4" per column baseline
            left_margin_in + (sec_cols * 2) + right_margin_in,
            4                                          # at least 4" total
        )

        # 5) Compute wspace so that there’s ~0.2" padding between columns
        total_axes_w = fig_w - left_margin_in - right_margin_in
        axes_w = total_axes_w / sec_cols
        desired_space_in = 0.2
        wspace = (desired_space_in / axes_w) + 1.3

        # 6) Create subplots (no constrained_layout)
        fig, axes = plt.subplots(
            nrows=1,
            ncols=sec_cols,
            figsize=(fig_w, fig_h),
            facecolor='black',
            constrained_layout=False
        )
        if sec_cols == 1:
            axes = [axes]
        else:
            axes = list(axes)

        # 7) Pad modules so total slots = sec_cols * max_rows
        items = list(modules.items())
        total_slots = sec_cols * max_rows
        padded = items + [("", 0)] * (total_slots - len(items))

        # 8) Plot each column
        for col_idx in range(sec_cols):
            ax = axes[col_idx]
            ax.set_facecolor('black')

            block = padded[col_idx*max_rows : (col_idx+1)*max_rows]
            labels, scores = zip(*block)
            fracs = [s/100.0 for s in scores]
            y = list(range(max_rows))

            ax.barh(y, fracs, color='#00FF00', edgecolor='white', height=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, color='white', fontproperties=label_fp, fontsize=8)
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

        # 9) Adjust margins: use left_frac/right_frac based on inches → fraction
        left_frac  = left_margin_in / fig_w
        right_frac = 1 - (right_margin_in / fig_w)

        fig.subplots_adjust(
            left=left_frac,
            right=right_frac,
            top=0.85,      # leave room for suptitle
            bottom=0.05,
            wspace=wspace
        )

        # 10) Add the section title centered at x=0.5 (figure coords)
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
    Generate a '90s video game'–style leaderboard PNG, 
    centering the title at the top of the figure (x=0.5).
    """
    # 1) Sort and take top_k
    sorted_lb = sorted(data.items(), key=lambda x: x[1], reverse=True)
    names  = [n for n, _ in sorted_lb][:top_k]
    scores = [s for _, s in sorted_lb][:top_k]

    # If empty, just write a title‐only image
    if not names:
        fig = plt.figure(figsize=(4, 2), facecolor='black')
        title_fp = register_font(font_path, size=20)
        fig.suptitle(
            "LEADERBOARD",
            x=0.5, y=0.5,
            ha='center', va='center',
            color='white',
            fontproperties=title_fp,
            fontsize=20
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
        plt.close(fig)
        return

    # 2) Compute fractions
    max_score = max(scores) if scores else 1
    fracs = [s / max_score for s in scores]

    # 3) Measure text widths for margins
    label_fp  = register_font(font_path, size=10)
    annot_fp  = register_font(font_path, size=10)
    title_fp  = register_font(font_path, size=20)

    # a) Longest name in inches
    max_name_w = max(get_text_width_inch(n, label_fp) for n in names)
    left_margin_in = max_name_w + 0.1

    # b) Largest score string (e.g. “142”)
    largest_str = str(max_score)
    annot_w = get_text_width_inch(largest_str, annot_fp)
    right_margin_in = annot_w + 0.1

    # 4) Figure size
    row_h = len(names) * 0.6 + 1     # 0.6" per row + 1" for title padding
    fig_h = max(row_h, 2)

    # Ensure at least 6" wide, and that bars get ~4" of space
    min_axes_w = 4
    fig_w = max(left_margin_in + min_axes_w + right_margin_in, 6)

    # 5) Convert margins to fraction
    left_frac  = left_margin_in / fig_w
    right_frac = 1 - (right_margin_in / fig_w)

    # 6) Create figure/axis (no tight_layout)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='black')
    ax.set_facecolor('black')

    # 7) Draw bars
    y = list(range(len(names)))
    ax.barh(y, fracs, color='#00FF00', edgecolor='white', height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color='white', fontproperties=label_fp, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])

    # 8) Annotate bars
    for i, (frac, sc) in enumerate(zip(fracs, scores)):
        sc_str = str(sc)
        # Decide inside vs. outside
        if frac >= 0.8:
            xtext = frac - (0.02)
            ha = 'right'
        else:
            xtext = frac + (0.02)
            ha = 'left'
        # Note: xtext is in *data* coords [0..1], which is fine
        ax.text(
            xtext, i, sc_str,
            va='center', ha=ha,
            color='white',
            fontproperties=annot_fp,
            fontsize=10
        )

    # 9) Centered title with fig.suptitle at x=0.5
    fig.suptitle(
        "LEADERBOARD",
        x=0.5, y=0.95,
        ha='center', va='top',
        color='white',
        fontproperties=title_fp,
        fontsize=20
    )

    # 10) Adjust subplot so margins aren’t clipped
    fig.subplots_adjust(
        left=left_frac,
        right=right_frac,
        top=0.90,    # leave space for the suptitle
        bottom=0.05
    )

    # 11) Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
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
            "some.super.long.module.name1": 60
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
