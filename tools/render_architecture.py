"""Regenerate the repository's architecture illustration (no empirical data)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/"docs/assets"
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 8), facecolor="#f3f6fa")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 14), ylim=(0, 8))
    ax.axis("off")
    navy, muted = "#11283f", "#596d82"
    teal, blue, orange = "#127f83", "#5366b1", "#bd702d"

    def text(x, y, value, size=11, color=navy, weight="normal", ha="left"):
        ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                ha=ha, va="center", family="DejaVu Sans")

    def box(x, y, w, h, title, detail, color, fill="#ffffff"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                   facecolor=fill, edgecolor="#d8e1eb", linewidth=1.1))
        ax.plot([x+.18, x+.18], [y+.2, y+h-.2], color=color, lw=3, solid_capstyle="round")
        text(x+.36, y+h-.34, title, 12, color, "bold")
        text(x+.36, y+.34, detail, 10, muted)

    def arrow(start, end, color=muted, style="arc3"):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                     lw=1.4, color=color, connectionstyle=style))

    text(.65, 7.35, "TSML–ChainPINN", 29, navy, "bold")
    text(.67, 6.87, "Transient cavitation  /  Progressive flow–phase learning", 13, muted)
    text(13.3, 7.35, "15.77M", 24, teal, "bold", ha="right")
    text(13.3, 6.91, "model parameters", 10, muted, ha="right")
    ax.plot([.65, 13.35], [6.5, 6.5], color="#d6e0ea", lw=1)

    box(.65, 4.23, 2.3, 1.2, "Space–time input", "(x, y, z, t) → normalize", navy)
    box(3.38, 4.23, 2.45, 1.2, "Fourier embedding", "64 bands · 2π encoding", teal)
    box(6.38, 5.04, 3.05, 1.12, "Flow branch", "10 × ResMLP · width 512", teal)
    box(6.38, 3.49, 3.05, 1.12, "Phase branch", "10 × ResMLP · width 512", blue)
    box(10.18, 4.23, 3.1, 1.2, "Coupled outputs", "p, u, v, w, αL  ·  αv = 1 − αL", orange)
    arrow((2.98, 4.83), (3.34, 4.83))
    arrow((5.86, 4.83), (6.34, 5.58), teal)
    arrow((5.86, 4.83), (6.34, 4.05), blue)
    arrow((9.46, 5.58), (10.15, 5.03), teal)
    arrow((9.46, 4.05), (10.15, 4.60), blue)
    arrow((7.52, 5.0), (7.52, 4.64), teal)
    arrow((8.32, 4.64), (8.32, 5.0), blue)
    text(7.92, 4.82, "↕", 13, navy, "bold", ha="center")
    text(7.91, 3.14, "Pointwise attention · 8 heads · feature exchange", 10, muted, ha="center")

    text(.65, 2.66, "THREE-STAGE OPTIMIZATION", 10, muted, "bold")
    stages = [(.65, "01  Flow pretraining", "80 epochs · lr 1e−3", "Flow L1 + divergence + momentum", teal),
              (4.98, "02  Phase learning", "120 epochs · lr 5e−4", "Focal + masked regression + FP penalty", blue),
              (9.31, "03  Joint fine-tuning", "160 epochs · lr 1e−4", "Coupled fields + ZGB vapor transport", orange)]
    for x, title, schedule, detail, color in stages:
        box(x, 1.05, 4.03, 1.3, title, detail, color)
        text(x+.36, 1.70, schedule, 10, muted)
    text(.67, .52, "Batch 4096  ·  αL < 0.90  ·  160 train / 40 test  ·  original-mesh inference", 11, navy)
    text(13.3, .52, "Method schematic", 9, muted, ha="right")
    fig.savefig(output/"architecture.png", dpi=160)
    fig.savefig(output/"architecture.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
