"""beautify — a custom, original publication-plot aesthetic for the SCOUT-FL report.

Not copied from tueplots / SciencePlots / BeautifulFigures — hand-built, with its own
opinions:
  * airy layout: only the two informative spines, offset from the data ("floating axes")
  * a soft dotted baseline grid, never competing with the data
  * a restrained, semantically-assigned palette (proposed = saturated, baselines = muted)
  * a signature "halo" glow behind the proposed method so the eye lands on it first
  * leader-line labels that never collide (used for dense scatter clouds)
  * quiet typographic hierarchy (light labels, one bold accent per panel)
All figures render to both .pdf (vector, paper) and .png (preview), Times New Roman.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke, Normal

# ── palette ───────────────────────────────────────────────────────────────────
INK   = "#000000"      # text — pure black (IEEE house style)
MUTE  = "#5b6169"      # secondary text only (NOT axes)
AXIS  = "#000000"      # axes, spines, ticks — complete black boxed axes
FAINT = "#c9ced4"      # light grid / hairlines
PAPER = "#ffffff"
PANEL = "#f7f8fa"      # soft panel wash

# semantic colours: the proposed method is saturated & warm-anchored; baselines muted
C = {
    "scout_v2":       "#0f5e8c",   # SCOUT-FL — deep confident blue (THE headline, only proposed method)
    "jedi":           "#c1263f",   # JEDI-FL — crimson (accuracy-max sibling)
    "scout_greedy":   "#5aa9a3",   # SCOUT-FL ablation (hard gate) — muted teal
    "collabsensefed": "#7d6ba7",   # strong baseline — muted violet
    "sensing_native": "#9c6b4e",   # strong baseline — clay
    "asaad":          "#e0912f",   # SOTA competitor (TWC'25) — amber
    "divfl":          "#6f9e57",   # learning-only — sage
    "random":         "#a6adb4",   # naive — grey
    "crb_only":       "#b7a83b",
    "sensing_only":   "#4babc4",
    "fedgcs":         "#c9a0c7",
    "oort":           "#b0895f",
}
def col(m): return C.get(m, "#b6bcc2")

# marker language: the proposed method = diamond, its sibling = star, ablation = circle
MARK = {"scout_v2": "D", "jedi": "*", "scout_greedy": "o"}
def mark(m): return MARK.get(m, "o")

HEADLINE = {"scout_v2"}                    # THE proposed method (only one now)
PROPOSED = {"scout_v2", "jedi", "scout_greedy"}   # for figures that still show the family

DISP = {
    "scout_v2": "SCOUT-FL", "jedi": "JEDI-FL",
    "scout_greedy": "SCOUT-FL (ablation: hard gate)",
    "collabsensefed": "CollabSenseFed", "sensing_native": "Sensing-Native",
    "asaad": "Asaad [TWC'25]", "divfl": "DivFL", "random": "Random",
    "crb_only": "CRB-Only", "sensing_only": "Sensing-Only", "fedgcs": "FedGCS", "oort": "Oort",
    "fair_equity": "FairEquityFL", "po_fl": "PO-FL", "delta": "DELTA", "fedis": "FedIS",
    "fedcs": "FedCS", "comm_only": "Comm-Only", "aircomp_mse_min": "AirComp-MSE-Min",
    "ota_fedavg": "OTA-FedAvg", "ota_fl_iscc": "OTA-FL-ISCC", "fed_iscc": "Fed-ISCC",
    "fedavg_iscc": "FedAvg-ISCC", "fedsgd_iscc": "FedSGD-ISCC", "iscc_air_feel": "ISCC-Air-FEEL",
    "fixed_weighted": "Fixed-Weighted", "loss": "Loss-Greedy", "snr_only": "SNR-Only",
}
def disp(m): return DISP.get(m, m)

SHORT = {   # compact labels for dense axes
    "scout_v2": "SCOUT-FL", "jedi": "JEDI-FL", "scout_greedy": "SCOUT-FL abl.",
    "collabsensefed": "CollabSenseFed", "sensing_native": "Sensing-Native",
    "asaad": "Asaad", "divfl": "DivFL", "random": "Random",
}
def short(m): return SHORT.get(m, disp(m))


def set_style():
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        # Figures are rendered at their EXACT print size (3.5in column / 7.16in text
        # width), so 8pt here is a true 8pt on the printed page (caption size).
        "font.size": 8.0, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.linewidth": 0.8, "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.labelsize": 8.0, "axes.labelweight": "normal",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK,
        "ytick.labelcolor": INK, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 340, "savefig.bbox": "tight",
        "axes.grid": False, "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def floating_axes(ax, x=True, y=True, offset=10):
    """Complete BLACK boxed axes: all four spines visible, joined, black — the standard
    IEEE frame. (Name kept for call-site compatibility; the old floating style is retired.)
    ``x``/``y`` False only suppresses that axis's ticks/labels, never the box."""
    for s in ("top", "right", "left", "bottom"):
        sp = ax.spines[s]
        sp.set_visible(True); sp.set_color(AXIS); sp.set_linewidth(1.1)
        sp.set_position(("outward", 0))
    ax.tick_params(which="both", length=4.0, width=1.0, colors=AXIS,
                   direction="out", top=False, right=False, labelcolor=INK)
    if not x: ax.tick_params(bottom=False, labelbottom=False)
    if not y: ax.tick_params(left=False, labelleft=False)


def soft_grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=FAINT, lw=0.6, ls="-", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def halo(ax, x, y, color, s=340):
    """Signature glow: a soft translucent ring behind a highlighted marker."""
    ax.scatter([x], [y], s=s*2.4, color=color, alpha=0.10, zorder=4, linewidths=0)
    ax.scatter([x], [y], s=s*1.5, color=color, alpha=0.16, zorder=4, linewidths=0)


def leader_label(ax, x, y, text, tx, ty, color=INK, weight="semibold", size=9.4,
                 ha="left", dot=True, lw=0.9):
    """Label placed at (tx,ty) in DATA coords, connected to (x,y) by a thin leader line —
    the dense-scatter-cloud-safe alternative to offset-point annotation (no overlap)."""
    if dot:
        ax.scatter([x], [y], s=18, color=color, zorder=6, linewidths=0)
    ax.plot([x, tx], [y, ty], color=color, lw=lw, alpha=0.55, zorder=5, solid_capstyle="round")
    ax.annotate(text, (tx, ty), ha=ha, va="center", color=color, fontsize=size,
                fontweight=weight, zorder=7,
                path_effects=[withStroke(linewidth=3.0, foreground="white"), Normal()])


def label_point(ax, x, y, text, color=INK, dx=8, dy=8, ha="left", va="bottom",
                weight="semibold", size=9.6):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va=va, color=color, fontsize=size, fontweight=weight, zorder=7,
                path_effects=[withStroke(linewidth=2.8, foreground="white"), Normal()])


def value_tag(ax, x, y, text, color, dx=0, dy=10, size=9.5, weight="bold", ha="center"):
    """Small pill-less value callout directly on a data element (bar tip, point, etc.)."""
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va="bottom", color=color, fontsize=size, fontweight=weight, zorder=8)


def panel_tag(ax, letter, x=-0.02, y=1.06):
    """(a) / (b) / (c) small-multiple tag, upper-left of the axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            color=INK, ha="left", va="bottom")


def title_block(fig, title, subtitle=None, x=0.012, y=0.99):
    fig.text(x, y, title, ha="left", va="top", fontsize=16, fontweight="bold", color=INK)
    if subtitle:
        fig.text(x, y-0.048, subtitle, ha="left", va="top", fontsize=10.2, color=MUTE)


def footnote(fig, text, x=0.012, y=0.006):
    fig.text(x, y, text, ha="left", va="bottom", fontsize=7.8, color=MUTE, style="italic")


def sig_marker(ax, x, y, significant, color, size=7.5):
    """Small asterisk/dagger badge for statistical significance next to a value."""
    txt = "†" if significant else ""
    if txt:
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=(3, 4),
                    color=color, fontsize=size, fontweight="bold", zorder=8)


def frame(fig, lw=1.2, pad=0.008):
    """Draw a complete black rectangular border around the whole figure — used for the
    schematic (axis-off) figures so they are 'inside a box' like the data plots."""
    from matplotlib.patches import Rectangle
    rect = Rectangle((pad, pad), 1 - 2 * pad, 1 - 2 * pad, transform=fig.transFigure,
                     fill=False, edgecolor=AXIS, linewidth=lw, zorder=1000, clip_on=False)
    fig.add_artist(rect)


def save(fig, path, box=False):
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    if box:
        frame(fig)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", pad_inches=0.06)
    plt.close(fig)
    print("  ●", os.path.basename(path))
