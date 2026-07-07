"""System-model diagrams for SCOUT-FL — grounded directly in the implementation
(scout_fl/objectives/total_utility.py, primal_dual.py, sim/fim.py, run_fl_synthetic.py).
Not decorative: every element corresponds to a real quantity in the code.

  fig_geometry  — the physical ISAC layer: BS, candidate/selected clients, targets,
                  AirComp uplink, bistatic sensing paths, and the anisotropic per-target
                  Fisher-information model (radial vs tangential decomposition).
  fig_pipeline  — the per-round algorithmic loop: probe -> composite submodular utility
                  -> greedy selection with the primal-dual AirComp-MSE penalty -> local
                  SGD -> over-the-air aggregation -> state update (coverage / fairness /
                  dual ascent) -> next round.

Run: python system_model.py   ->  figures/fig1_system.{pdf,png}, figures/fig12_algorithm.{pdf,png}
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse
from matplotlib.path import Path as MPath
import matplotlib.patches as mpatches

import beautify as bu

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
bu.set_style()

BLUE = bu.C["scout_v2"]; RED = bu.C["jedi"]; MUTE = bu.MUTE; INK = bu.INK
GREY = "#c9cdd2"; GOLD = "#c99a2e"; GREEN = "#5aa06a"


# ═══════════════════════════════════════════════════════════ FIG A — ISAC geometry
def fig_geometry():
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(11.2, 7.4))
    ax = fig.add_axes([0.03, 0.05, 0.64, 0.88])
    axz = fig.add_axes([0.68, 0.30, 0.30, 0.46])   # zoom inset: FIM decomposition

    # ---- main panel: cell layout ----
    R = 1.0
    bs = np.array([0.0, 0.0])
    cell = Circle(bs, R, fill=False, ls=(0, (4, 3)), lw=1.3, color=MUTE, zorder=1)
    ax.add_patch(cell)
    ax.text(0, R * 1.06, "cell boundary", ha="center", color=MUTE, fontsize=9, style="italic")

    K = 26
    ang = rng.uniform(0, 2 * np.pi, K)
    rad = np.sqrt(rng.uniform(0.08, 0.97, K)) * R
    clients = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    selected_idx = [2, 7, 12, 18, 22]

    targets = np.array([[0.35, 0.55], [-0.5, -0.25], [0.55, -0.45]])
    tgt_names = ["target $m_1$", "target $m_2$", "target $m_3$"]

    # sensing paths: every selected client -> every target (bistatic illumination + return)
    for si in selected_idx:
        for t in targets:
            ax.plot([clients[si, 0], t[0]], [clients[si, 1], t[1]], color=GOLD,
                    lw=0.8, ls=(0, (1, 2)), alpha=0.55, zorder=2)
    # AirComp uplink: selected clients -> BS (superposed, same slot -> one thick braided arrow look)
    for si in selected_idx:
        ax.add_patch(FancyArrowPatch(clients[si], bs * 0.06 + clients[si] * 0.0 + bs,
                                     arrowstyle="-|>", mutation_scale=12, color=BLUE,
                                     lw=1.8, alpha=0.85, zorder=4,
                                     connectionstyle="arc3,rad=0.0",
                                     shrinkA=8, shrinkB=14))

    # candidate (unselected) clients
    others = [i for i in range(K) if i not in selected_idx]
    ax.scatter(clients[others, 0], clients[others, 1], s=60, color=GREY, edgecolor="white",
              lw=0.8, zorder=5, label="candidate client (not selected)")
    # selected clients
    ax.scatter(clients[selected_idx, 0], clients[selected_idx, 1], s=140, color=BLUE,
              edgecolor="white", lw=1.4, zorder=6, marker="D", label="selected client  $k \\in S_t$")

    # BS
    ax.scatter(*bs, s=520, color=INK, marker="^", zorder=7, edgecolor="white", lw=1.6)
    ax.text(bs[0], bs[1] - 0.115, "base station\n(edge server)", ha="center", va="top",
            fontsize=9.3, fontweight="bold", color=INK, zorder=7)

    # targets
    for t, name in zip(targets, tgt_names):
        ax.scatter(*t, s=230, color=RED, marker="*", zorder=6, edgecolor="white", lw=1.2)
        ax.annotate(name, t, xytext=(9, 7), textcoords="offset points", fontsize=9.2,
                    color=RED, fontweight="semibold")

    # zoom-region indicator around client[selected_idx[0]] <-> targets[0]
    focus_c, focus_t = clients[selected_idx[0]], targets[0]
    for pt in (focus_c, focus_t):
        ax.add_patch(Circle(pt, 0.045, fill=False, color=INK, lw=1.0, zorder=8))

    ax.set_xlim(-1.28, 1.28); ax.set_ylim(-1.2, 1.28)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="lower left", fontsize=9.2, frameon=False,
             handles=[
                 mpatches.Patch(color=GREY, label="candidate client (not selected)"),
                 mpatches.Patch(color=BLUE, label="selected client  $k \\in S_t$  ($|S_t|=K$)"),
                 mpatches.Patch(color=RED, label="physical target  $m$"),
                 mpatches.Patch(color=BLUE, label="AirComp uplink (superposed, one slot)"),
                 mpatches.Patch(color=GOLD, label="bistatic sensing path (illumination + return)"),
             ], handlelength=1.4, labelspacing=0.55)

    # ---- inset: anisotropic per-client-target Fisher information (simple horizontal
    # layout so every label has unambiguous, non-overlapping room) ----
    axz.set_xlim(-0.30, 1.75); axz.set_ylim(-0.95, 0.95); axz.axis("off")
    c0 = np.array([0.0, 0.0]); t0 = np.array([1.35, 0.0])         # client -> target, purely radial
    u = np.array([1.0, 0.0]); v = np.array([0.0, 1.0])            # radial / tangential unit vectors

    # information ellipse drawn FIRST (behind the markers): long axis along u (range term
    # a_r), short axis along v (angle term a_a) — axes lengths ∝ information, not uncertainty
    ell = Ellipse(t0, width=0.85, height=0.30, angle=0, facecolor=GOLD, alpha=0.30,
                 edgecolor=GOLD, lw=1.3, zorder=2)
    axz.add_patch(ell)

    axz.add_patch(FancyArrowPatch(c0, t0 - np.array([0.06, 0]), arrowstyle="-|>", mutation_scale=15,
                                  color=INK, lw=1.7, zorder=5))
    axz.text(t0[0] / 2, 0.10, "$u$ (radial / range)", fontsize=9.0, color=INK, ha="center", va="bottom")
    axz.text(t0[0] / 2, -0.10, "$a_r = \\gamma\\,k_{\\mathrm{range}}$", fontsize=8.4, color=INK,
            ha="center", va="top")

    v_base = c0 + u * 0.30
    axz.add_patch(FancyArrowPatch(v_base, v_base + v * 0.55, arrowstyle="-|>", mutation_scale=13,
                                  color=GREEN, lw=1.7, zorder=5))
    axz.text(v_base[0] - 0.06, v_base[1] + 0.58, "$v$ (tangential / angle)",
            fontsize=8.8, color=GREEN, ha="right", va="center")
    axz.text(v_base[0] - 0.06, v_base[1] + 0.36, "$a_a = \\gamma\\,k_{\\mathrm{angle}}/r^2$",
            fontsize=8.2, color=GREEN, ha="right", va="center")

    axz.scatter(*c0, s=150, color=BLUE, marker="D", zorder=6, edgecolor="white", lw=1.3)
    axz.text(c0[0], c0[1] - 0.20, "client $k$", fontsize=9.4, color=BLUE, ha="center",
            va="top", fontweight="semibold")
    axz.scatter(*t0, s=210, color=RED, marker="*", zorder=6, edgecolor="white", lw=1.3)
    axz.text(t0[0], t0[1] + 0.42, "target $m$", fontsize=9.4, color=RED, ha="center",
            va="bottom", fontweight="semibold")

    axz.text(t0[0] / 2, -0.80, r"$J_{k,m} = a_r\,uu^{\!\top} + a_a\,vv^{\!\top}$",
             fontsize=11.2, ha="center", color=INK)
    axz.set_title("per-client, per-target Fisher information\n(anisotropic — bearing sets the axis)",
                  fontsize=9.8, color=MUTE, fontweight="normal", pad=6)

    bu.title_block(fig, "SCOUT-FL system model: the ISAC network",
                   "A base station selects $K$ of $N$ clients per round for joint AirComp model aggregation and multistatic target sensing")
    bu.footnote(fig, "Two clients at similar bearings pile information onto the same axis of $J_{k,m}$ (diminishing sensing return); angular diversity across selected clients — not raw SNR — drives $\\log\\det$ information gain.")
    bu.save(fig, os.path.join(FIG, "fig1_system"), box=True)


# ═══════════════════════════════════════════════════════════ FIG B — algorithm pipeline
def _hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def _box(ax, xy, w, h, text, color, fc_alpha=0.10, fontsize=9.6, weight="bold", tcolor=None):
    """Rounded box with a translucent fill and a solid border. NOTE: Patch.alpha (if set)
    overrides per-channel facecolor alpha in matplotlib, so we deliberately leave the
    patch-level alpha at its default (None) and encode translucency only in the RGBA
    facecolor tuple — otherwise the box renders fully opaque regardless of fc_alpha."""
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                       linewidth=1.6, edgecolor=color, facecolor=(*_hex2rgb(color), fc_alpha),
                       zorder=4)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
           fontsize=fontsize, fontweight=weight, color=tcolor or color, zorder=6, linespacing=1.4)
    return dict(cx=xy[0] + w / 2, cy=xy[1] + h / 2, l=xy[0], r=xy[0] + w, t=xy[1] + h, b=xy[1], w=w, h=h)


def _hconnect(ax, a, b_, color=INK, lw=1.8):
    """Straight edge-to-edge arrow: right edge of box a -> left edge of box b_."""
    ax.add_patch(FancyArrowPatch((a["r"], a["cy"]), (b_["l"], b_["cy"]), arrowstyle="-|>",
                                 mutation_scale=16, color=color, lw=lw, zorder=5,
                                 shrinkA=1, shrinkB=1))


def _vconnect(ax, a, b_, color=INK, lw=1.8):
    ax.add_patch(FancyArrowPatch((a["cx"], a["b"]), (b_["cx"], b_["t"]), arrowstyle="-|>",
                                 mutation_scale=16, color=color, lw=lw, zorder=5,
                                 shrinkA=1, shrinkB=1))


def _elbow_loop(ax, start, end, y_rail, color=INK, lw=1.7, ls="-", mutation_scale=15):
    """Manual 3-segment (up / across / down) feedback path — robust for long loops where
    a smooth Bezier arc (arc3) would either barely bow or overshoot unpredictably."""
    x0, y0 = start; x1, y1 = end
    path = MPath([(x0, y0), (x0, y_rail), (x1, y_rail), (x1, y1 + 0.22)],
                [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO])
    ax.add_patch(mpatches.PathPatch(path, facecolor="none", edgecolor=color, lw=lw,
                                    linestyle=ls, zorder=5, capstyle="round"))
    ax.add_patch(FancyArrowPatch((x1, y1 + 0.22), (x1, y1), arrowstyle="-|>", mutation_scale=mutation_scale,
                                 color=color, lw=lw, zorder=5, linestyle="-"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    ax.set_xlim(0, 12.8); ax.set_ylim(0.75, 6.6); ax.axis("off")

    y = 3.35; h = 1.55; w = 1.92; gap = 0.46
    xs = [0.35 + i * (w + gap) for i in range(5)]

    p1 = _box(ax, (xs[0], y), w, h,
              "1.  Probe\n\nbroadcast $w_t$;\nlocal loss + grad.\nembedding $e_k$\n(all $N$ clients)",
              MUTE, fontsize=8.9, weight="normal", tcolor=INK)
    p2 = _box(ax, (xs[1], y), w, h,
              "2.  Composite utility\n\n$f_{\\mathrm{learn}}\\!+\\!f_{\\mathrm{sense}}$\n$+f_{\\mathrm{cov}}\\!+\\!f_{\\mathrm{fair}}$\n(submodular sum)",
              GREEN, fontsize=9.0, weight="normal", tcolor=INK)
    p3 = _box(ax, (xs[2], y), w, h,
              "3.  Greedy select\n\n$\\arg\\max\\ \\Delta U(k)$\n$-\\ \\mu\\cdot$MSE-penalty\ns.t.  $|S_t| = K$",
              BLUE, fontsize=9.0, weight="normal", tcolor=INK)
    p4 = _box(ax, (xs[3], y), w, h,
              "4.  Local SGD\n\nclients $k \\in S_t$\ntrain on local data\n(FedAvg update)",
              MUTE, fontsize=8.9, weight="normal", tcolor=INK)
    p5 = _box(ax, (xs[4], y), w, h,
              "5.  AirComp aggregate\n\nchannel-inversion\nOTA sum,  MSE$(S_t)$",
              GOLD, fontsize=8.9, weight="normal", tcolor=INK)

    for a, b_ in [(p1, p2), (p2, p3), (p3, p4), (p4, p5)]:
        _hconnect(ax, a, b_)

    # step 6: evaluate + update state (below, wide box)
    y6 = 1.05; h6 = 1.15
    p6 = _box(ax, (xs[1] - 0.15, y6), (xs[4] + w) - (xs[1] - 0.15), h6,
              "6.  Evaluate & update state:   $w_{t+1}$   ·   coverage map   ·   fairness ages   ·   "
              "dual ascent   $\\mu \\leftarrow [\\,\\mu + \\eta\\,(\\mathrm{MSE}(S_t) - \\varepsilon)\\,]_+$",
              INK, fc_alpha=0.055, fontsize=10.2, weight="normal", tcolor=INK)

    _vconnect(ax, p5, p6)

    # feedback #1: state update -> next round's probe (wide loop over the whole row)
    _elbow_loop(ax, (p6["l"] + 0.35, p6["cy"]), (p1["cx"], p1["t"]), y_rail=6.05, color=INK, lw=1.7)
    ax.text((p6["l"] + 0.35 + p1["cx"]) / 2, 6.20, "next round   $t \\leftarrow t+1$",
           fontsize=10.0, color=INK, fontweight="semibold", ha="center")

    # feedback #2: dual ascent specifically re-enters step 3's penalty — box 6 sits directly
    # under box 3, so this is a short straight dashed riser (no elbow needed), labelled beside it
    mu_x = p3["cx"] - 0.5
    ax.add_patch(FancyArrowPatch((mu_x, p6["t"]), (mu_x, p3["b"]), arrowstyle="-|>",
                                 mutation_scale=14, color=BLUE, lw=1.9, zorder=5,
                                 linestyle=(0, (5, 2.5)), shrinkA=1, shrinkB=1))
    ax.text(mu_x - 0.14, (p6["t"] + p3["b"]) / 2, "primal–dual\nfeedback\n(v2 only):\n$\\mu$ reprices\nAirComp-weak\nclients",
           fontsize=7.9, color=BLUE, ha="right", va="center", fontweight="semibold", style="italic",
           linespacing=1.3)

    bu.title_block(fig, "SCOUT-FL: the per-round selection–training–aggregation loop",
                   "Every quantity is real: $f_{\\mathrm{sense}}$ = weighted log-det FIM gain (sim/fim.py); the AirComp-MSE constraint is enforced softly via dual ascent, not a hard gate (the ablation lacks this)")
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.01, right=0.99)
    bu.footnote(fig, "Boxes 1–5 execute once per communication round; the dashed loop is the soft primal-dual constraint that SCOUT-FL's hard-gate ablation lacks.")
    bu.save(fig, os.path.join(FIG, "fig12_algorithm"))


# ═══════════════════════════════════════════ FIG 1 — block-style system model (10pt)
def fig_blocks():
    """Paper Fig. 1: the SCOUT-FL ISAC-FL loop as a block diagram, drawn at the exact
    \\textwidth print size (7.16 in) so every label is a true 10 pt — the same size as
    the manuscript body text. Same content as the old draw.io sys_model.png."""
    W, H = 7.16, 3.75
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_aspect("equal")
    LAV = "#7d6ba7"; GRN = "#2a7d4f"; NOIR = INK

    def panel(x0, y0, x1, y1, label, color=MUTE):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                     boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.0,
                     edgecolor=color, facecolor="#f7f8fa", linestyle=(0, (4, 3)), zorder=1))
        ax.text((x0 + x1) / 2, y1 - 0.13, label, ha="center", va="top", fontsize=10,
                fontweight="bold", color=INK, zorder=3)

    def block(x0, y0, w, h, lines, color, lw=1.4, fs=10):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                     boxstyle="round,pad=0.015,rounding_size=0.05", linewidth=lw,
                     edgecolor=color, facecolor=(*_hex2rgb(color), 0.10), zorder=4))
        ax.text(x0 + w / 2, y0 + h / 2, lines, ha="center", va="center", fontsize=fs,
                color=INK, zorder=6, linespacing=1.45)
        return dict(cx=x0 + w / 2, cy=y0 + h / 2, l=x0, r=x0 + w, t=y0 + h, b=y0)

    def arrow(p0, p1, color=NOIR, lw=1.6, ls="-", rad=0.0, ms=13):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                     color=color, lw=lw, linestyle=ls, zorder=5,
                     connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))

    # ── left panel: active clients ────────────────────────────────────────────
    panel(0.08, 0.30, 2.02, 3.28, "active clients $k \\in \\mathcal{S}_t$")
    c1 = block(0.22, 2.08, 1.66, 0.80,
               "client $1$\nSGD $\\Delta \\mathbf{w}_{1,t}$ on $\\mathcal{D}_1$\n"
               "precode $b_{1,t}=\\rho_t/h_{1,t}$", BLUE, lw=1.2)
    ax.text(1.05, 1.88, "$\\vdots$", ha="center", va="center", fontsize=12, color=INK)
    cK = block(0.22, 0.85, 1.66, 0.80,
               "client $K$\nSGD $\\Delta \\mathbf{w}_{K,t}$ on $\\mathcal{D}_K$\n"
               "precode $b_{K,t}=\\rho_t/h_{K,t}$", BLUE, lw=1.2)

    # ── middle: AirComp MAC node + target ─────────────────────────────────────
    mac = (2.62, 2.48)
    ax.add_patch(Circle(mac, 0.20, facecolor=(*_hex2rgb(BLUE), 0.12), edgecolor=BLUE,
                        lw=1.5, zorder=5))
    ax.text(*mac, "$\\Sigma$", ha="center", va="center", fontsize=13, color=BLUE,
            fontweight="bold", zorder=6)
    ax.text(mac[0], mac[1] - 0.30, "AirComp\nMAC", ha="center", va="top", fontsize=10,
            color=INK, style="italic", linespacing=1.2)
    # uplink noise
    arrow((mac[0], 3.42), (mac[0], mac[1] + 0.22), color=RED, lw=1.4)
    ax.text(mac[0] + 0.08, 3.44, "$\\mathbf{n}_t \\sim \\mathcal{N}(0,\\sigma^2\\mathbf{I})$",
            ha="left", va="center", fontsize=10, color=RED)
    # target
    tgt = (2.62, 1.05)
    ax.add_patch(Circle(tgt, 0.17, facecolor=(*_hex2rgb(GOLD), 0.25), edgecolor=GOLD,
                        lw=1.5, zorder=5))
    ax.text(*tgt, "$m$", ha="center", va="center", fontsize=10, color=INK,
            fontweight="bold", zorder=6)
    ax.text(tgt[0], tgt[1] - 0.26, "target", ha="center", va="top", fontsize=10,
            color=GOLD, style="italic")

    # uplink arrows: clients -> MAC (solid blue, superposed same slot)
    arrow((c1["r"], c1["cy"]), (mac[0] - 0.21, mac[1] + 0.06), color=BLUE, lw=1.8, rad=-0.06)
    arrow((cK["r"], cK["cy"] + 0.15), (mac[0] - 0.20, mac[1] - 0.12), color=BLUE, lw=1.8, rad=0.18)
    # sensing: illumination client K -> target (dashed gold), echo target -> radar box
    arrow((cK["r"], cK["cy"] - 0.15), (tgt[0] - 0.19, tgt[1]), color=GOLD, lw=1.5,
          ls=(0, (4, 2)), rad=0.10)
    ax.text(2.22, 0.48, "illumination", ha="center", fontsize=10, color=GOLD, style="italic")

    # ── right panel: BS / edge server ─────────────────────────────────────────
    panel(3.30, 0.30, 7.08, 3.28, "base station / edge server")
    agg = block(3.44, 2.08, 1.98, 0.80,
                "RF reception + AirComp\n$\\mathbf{y}_t=\\sum_k h b\\,\\Delta\\mathbf{w}_k+\\mathbf{n}_t$\n"
                "$\\mathbf{w}_{t+1}=\\mathbf{w}_t+\\mathbf{y}_t$", GRN, lw=1.4)
    rad = block(3.44, 0.55, 1.98, 0.80,
                "multistatic radar estimator\n$\\mathbf{J}_m(\\mathcal{S})=\\mathbf{J}_0+"
                "\\sum_k \\mathbf{J}_{k,m}$", GOLD, lw=1.4)
    sel = block(5.62, 0.55, 1.32, 2.33,
                "SCOUT-FL\nprimal–dual\nselector\n\ngreedy max\n$U_{\\mathrm{lrn}}"
                "{+}\\lambda\\Sigma\\log\\det\\mathbf{J}_m$\n\n$\\mu{\\leftarrow}[\\mu{+}\\eta("
                "\\mathrm{MSE}{-}\\varepsilon)]_+$\n\nnext set $\\mathcal{S}_{t+1}$",
                BLUE, lw=1.8)

    arrow((mac[0] + 0.21, mac[1]), (agg["l"] - 0.02, agg["cy"]), color=BLUE, lw=1.8)
    arrow((tgt[0] + 0.18, tgt[1]), (rad["l"] - 0.02, rad["cy"]), color=GOLD, lw=1.5,
          ls=(0, (4, 2)))
    ax.text(3.02, 0.72, "echo", ha="center", fontsize=10, color=GOLD, style="italic")
    arrow((agg["r"], agg["cy"]), (sel["l"], agg["cy"]), color=NOIR, lw=1.5)
    arrow((rad["r"], rad["cy"]), (sel["l"], rad["cy"]), color=NOIR, lw=1.5)

    # ── broadcast downlink: selector -> clients (red dashed elbow over the top) ──
    yr = 3.62
    path = MPath([(sel["cx"], sel["t"]), (sel["cx"], yr), (1.05, yr), (1.05, c1["t"] + 0.10)],
                 [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO])
    ax.add_patch(mpatches.PathPatch(path, facecolor="none", edgecolor=RED, lw=1.5,
                                    linestyle=(0, (5, 3)), zorder=5, capstyle="round"))
    ax.add_patch(FancyArrowPatch((1.05, c1["t"] + 0.10), (1.05, c1["t"]), arrowstyle="-|>",
                                 mutation_scale=13, color=RED, lw=1.5, zorder=5))
    ax.text(4.55, yr + 0.02, "broadcast downlink  $(\\mathbf{w}_{t+1},\\,\\mathcal{S}_{t+1})$",
            ha="center", va="bottom", fontsize=10, color=RED, fontweight="semibold")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    bu.save(fig, os.path.join(FIG, "fig1_blocks"))


def main():
    print("Rendering system-model diagrams:")
    fig_blocks()
    print(f"Done -> {FIG}")


if __name__ == "__main__":
    main()
