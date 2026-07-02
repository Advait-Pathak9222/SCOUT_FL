"""Shared data-loading + method taxonomy for the preliminary SCOUT-FL report.

Loads every per-(point, method, seed) JSON under runs/<tag>/ into a tidy DataFrame
using the same final-round-snapshot convention as scout_fl.analysis.collect, and
attaches human-readable names / categories / citations for each selection method.
"""
from __future__ import annotations
import json, glob, os
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(REPO, "runs")

# --------------------------------------------------------------------------- #
# Method taxonomy.  is_proposed => one of OUR methods (never a baseline).
# group is used to colour/organise tables and plots.
# --------------------------------------------------------------------------- #
# NOTE: per project decision, JEDI-FL and VISMAYA-FL are the SAME proposed method.
# It is represented by the tuned `jedi` configuration; the separate `vismaya` run is
# an (untuned) generative-synergy variant kept only inside the mobility ablation.
PROPOSED = {"jedi", "scout_v2", "scout_greedy"}

META = {
    # ---- proposed ----------------------------------------------------------
    "jedi":            ("JEDI/VISMAYA-FL (ours)", "Proposed", "joint experimental-design + generative innovation"),
    "scout_v2":        ("SCOUT-FL v2 (ours)",    "Proposed", "submodular + primal-dual MSE"),
    "scout_greedy":    ("SCOUT-FL v1 (ours)",    "Proposed", "submodular + hard MSE gate"),
    "vismaya":         ("VISMAYA gen.\\ variant","Ablation", "innovation-driven generative (untuned)"),
    # ---- ISAC / sensing-aware baselines -----------------------------------
    "asaad":           ("Asaad et al. [TWC'25]", "ISAC",     "sensing-aware OTA-FEEL (MSE+CRB drop)"),
    "collabsensefed":  ("CollabSenseFed",        "ISAC",     "multi-objective learn+CRB (equal w)"),
    "fed_iscc":        ("Fed-ISCC [IoT-J'24]",   "ISAC",     "joint sense+OTA-FL, SNR-ranked"),
    "ota_fl_iscc":     ("OTA-FL-ISCC [ICC-W'24]","ISAC",     "AirComp-MSE-gated learning select"),
    "sensing_native":  ("Sensing-Native OTA-FL", "ISAC",     "grad-reused sensing, learn+sense"),
    "iscc_air_feel":   ("ISCC-Air-FEEL [2025]",  "ISAC",     "sensing-noise x channel restriction"),
    "fedavg_iscc":     ("FedAvg-ISCC",           "ISAC",     "ISCC resource select (FedAvg)"),
    "fedsgd_iscc":     ("FedSGD-ISCC",           "ISAC",     "ISCC resource select (FedSGD)"),
    "fixed_weighted":  ("Fixed-Weighted ISAC",   "ISAC",     "hand-set a*learn+b*sense-c*MSE"),
    # ---- sensing / comm primitives ----------------------------------------
    "sensing_only":    ("Sensing-Only (D-opt)",  "Sensing",  "max log-det Fisher info"),
    "crb_only":        ("CRB-Only (A-opt)",      "Sensing",  "min aggregate CRB"),
    "aircomp_mse_min": ("AirComp-MSE-Min",       "Comm",     "min AirComp aggregation MSE"),
    "comm_only":       ("Comm-Only",             "Comm",     "strongest uplink channels"),
    "snr_only":        ("SNR-Only",              "Comm",     "highest sensing SNR"),
    # ---- learning-driven FL selection -------------------------------------
    "oort":            ("Oort [OSDI'21]",        "Learning", "loss x speed utility"),
    "fedis":           ("FedIS",                 "Learning", "gradient-norm importance sampling"),
    "po_fl":           ("PO-FL",                 "Learning", "channel x gradient importance"),
    "divfl":           ("DivFL [ICLR'22]",       "Learning", "submodular gradient diversity"),
    "delta":           ("DELTA [NeurIPS'23]",    "Learning", "unbiased diverse selection"),
    "fair_equity":     ("FairEquityFL [2025]",   "Learning", "sampling-equalizer fairness"),
    "fedgcs":          ("FedGCS [IJCAI'24]",     "Learning", "diversity vs execution cost"),
    "loss":            ("Loss-Greedy",           "Learning", "highest local loss"),
    "fedcs":           ("FedCS [2019]",          "Learning", "fastest feasible clients"),
    # ---- naive -------------------------------------------------------------
    "random":          ("Random",               "Naive",    "uniform random-K"),
    "ota_fedavg":      ("OTA-FedAvg (Rand-K)",   "Naive",    "random-K + AirComp"),
    # ---- ablation variants (excluded from the main bake-off table) --------
    "vismaya_no_syn":     ("VISMAYA w/o synergy",  "Ablation", "beta=0"),
    "vismaya_sense_only": ("VISMAYA sense-only",   "Ablation", "learning term off"),
    "vismaya_learn_only": ("VISMAYA learn-only",   "Ablation", "sensing term off"),
    "jedi_no_fairness":   ("JEDI w/o fairness",    "Ablation", "deficit term off"),
    "jedi_no_coverage":   ("JEDI w/o coverage",    "Ablation", "coverage term off"),
    "jedi_no_sensing":    ("JEDI w/o sensing",     "Ablation", "sensing term off"),
    "jedi_no_learning":   ("JEDI w/o learning",    "Ablation", "learning term off"),
    "jedi_no_externality":("JEDI w/o externality", "Ablation", "MSE externality off"),
    "jedi_no_kappa":      ("JEDI w/o kappa",       "Ablation", "coupling off"),
    "jedi_fixed_rho":     ("JEDI fixed-rho",       "Ablation", "no auto-calibration"),
    "jedi_hard_gate":     ("JEDI hard-gate",       "Ablation", "SCOUT-v1 MSE gate"),
    "jedi_twin":          ("JEDI + twin",          "Ablation", "trust-gated residual twin"),
}

GROUP_ORDER = ["Proposed", "ISAC", "Learning", "Sensing", "Comm", "Naive", "Ablation"]


def disp(method: str) -> str:
    return META.get(method, (method, "Other", ""))[0]

def group(method: str) -> str:
    return META.get(method, (method, "Other", ""))[1]

def descr(method: str) -> str:
    return META.get(method, (method, "Other", ""))[2]

def is_proposed(method: str) -> bool:
    return method in PROPOSED


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
_OBJ_KEYS = ["acc", "best_acc", "logdet", "crb", "agg_mse", "jain", "energy",
             "round_s", "logdet_final", "crb_final"]


def load_tag(tag: str) -> pd.DataFrame:
    """Return one row per complete (point, method, seed) run under runs/<tag>/."""
    rows = []
    for f in glob.glob(os.path.join(RUNS, tag, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not d.get("complete", False):
            continue
        m = d.get("meta", {})
        o = d.get("objectives", {})
        row = {"tag": tag, "point": m.get("point", "base"), "method": m.get("method"),
               "seed": m.get("seed"), "dataset": m.get("dataset"), "model": m.get("model"),
               "rounds": m.get("rounds")}
        for k in _OBJ_KEYS:
            row[k] = o.get(k, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def load_rounds(tag: str, point: str = "base") -> pd.DataFrame:
    """Per-round trajectories for a tag/point (used for convergence curves)."""
    rows = []
    base = os.path.join(RUNS, tag)
    pdir = os.path.join(base, point)
    search = pdir if os.path.isdir(pdir) else base
    for f in glob.glob(os.path.join(search, "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not d.get("complete", False):
            continue
        m = d.get("meta", {})
        if m.get("point", "base") != point:
            continue
        for r in d.get("rounds", []):
            rows.append({"method": m["method"], "seed": m["seed"],
                         "round": r["round"], "test_acc": r.get("test_acc"),
                         "sensing_logdet": r.get("sensing_logdet"), "crb": r.get("crb"),
                         "vis_synergy_mean": r.get("vis_synergy_mean")})
    return pd.DataFrame(rows)


def agg_over_seeds(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Mean/std over seeds per (point, method)."""
    if metrics is None:
        metrics = ["acc", "best_acc", "logdet", "crb", "agg_mse", "jain", "energy", "round_s"]
    g = df.groupby(["point", "method"])
    out = g[metrics].agg(["mean", "std", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    return out


# --------------------------------------------------------------------------- #
# Pareto helpers.  Convention: we maximise acc and minimise crb (so we flip crb).
# For (acc, logdet) both are maximised.
# --------------------------------------------------------------------------- #
def pareto_mask(xs, ys, x_max=True, y_max=True):
    """Return boolean mask of Pareto-optimal points."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    sx = xs if x_max else -xs
    sy = ys if y_max else -ys
    n = len(xs)
    keep = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (sx[j] >= sx[i] and sy[j] >= sy[i]) and (sx[j] > sx[i] or sy[j] > sy[i]):
                keep[i] = False
                break
    return keep


def minmax_norm(v):
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


if __name__ == "__main__":
    for tag in ["campaign_main", "campaign", "ablation", "ablation_vismaya"]:
        df = load_tag(tag)
        print(f"{tag:16s} rows={len(df):5d} points={df['point'].nunique():3d} "
              f"methods={df['method'].nunique():3d}")
