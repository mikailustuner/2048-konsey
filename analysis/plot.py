"""Plots and tables from the C++ tools' JSONL logs.

  uv run --project analysis analysis/plot.py curve  runs/faz2/train.jsonl [more.jsonl ...] --out curve.png
  uv run --project analysis analysis/plot.py table  runs/faz3/eval.jsonl  [--faz 3]
  uv run --project analysis analysis/plot.py evolve runs/faz4/generations.jsonl --out evolve.png
"""
import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette (dataviz skill), light mode; categorical slots in fixed order.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TILES = ["2048", "4096", "8192", "16384", "32768", "65536"]


def read(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ax.spines.values():
        s.set_visible(False)


def curve(args):
    """Learning curve: mean score and tile reach rates vs training games (small multiples, one axis each)."""
    rows = read(args.logs)
    runs = defaultdict(list)
    for r in rows:
        if "egitim_oyunu" in r:
            runs[f'{r["run_id"]}/{r["ajan_id"]}'].append(r)
    tiles = ["2048", "8192", "16384", "32768"]
    fig, axes = plt.subplots(1 + len(tiles), 1, figsize=(8, 2.2 * (1 + len(tiles))), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for i, (run, rs) in enumerate(runs.items()):
        rs.sort(key=lambda r: r["egitim_oyunu"])
        x = [r["egitim_oyunu"] for r in rs]
        c = SERIES[i % len(SERIES)]
        axes[0].plot(x, [r["ort_skor"] for r in rs], color=c, linewidth=2, label=run)
        for ax, t in zip(axes[1:], tiles):
            ax.plot(x, [100 * r["oranlar"][t] for r in rs], color=c, linewidth=2)
    style(axes[0], "Mean score (training games, window mean)", "score")
    for ax, t in zip(axes[1:], tiles):
        style(ax, f"{t} reach rate", "%")
    axes[-1].set_xlabel("training games", color=INK2, fontsize=9)
    if len(runs) > 1:
        axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print("written:", args.out)


def table(args):
    """Markdown table of evaluation lines (no egitim_oyunu)."""
    rows = [r for r in read(args.logs) if "egitim_oyunu" not in r and (args.faz is None or r["faz"] == args.faz)]
    head = ["agent", "games", "mean score"] + TILES[:5]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        cells = [r["ajan_id"], str(r["oyun_sayisi"]), f'{r["ort_skor"]:,.0f}']
        for t in TILES[:5]:
            lo, hi = r["guven"][t]
            cells.append(f'{100 * r["oranlar"][t]:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]')
        print("| " + " | ".join(cells) + " |")


def evolve(args):
    """Per-generation best and mean tournament score."""
    rows = read(args.logs)
    gens = defaultdict(list)
    for r in rows:
        gens[r["nesil"]].append(r["ort_skor"])
    g = sorted(gens)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(SURFACE)
    ax.plot(g, [max(gens[i]) for i in g], color=SERIES[0], linewidth=2, label="best agent")
    ax.plot(g, [sum(gens[i]) / len(gens[i]) for i in g], color=SERIES[1], linewidth=2, label="population mean")
    style(ax, "Tournament score over generations", "mean score")
    ax.set_xlabel("generation", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print("written:", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["curve", "table", "evolve"])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="plot.png")
    ap.add_argument("--faz")
    args = ap.parse_args()
    {"curve": curve, "table": table, "evolve": evolve}[args.cmd](args)


if __name__ == "__main__":
    main()
