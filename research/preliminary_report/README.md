# SCOUT-FL / JEDI(VISMAYA)-FL — Preliminary Report + Publishability Verdict

Self-contained IEEE-style LaTeX package comparing the two proposed sensing-aware
client-selection families — **SCOUT-FL** (v1/v2) and **JEDI/VISMAYA-FL** — against
25 baselines, generated directly from the run artifacts in `../../runs/`.

> **JEDI-FL and VISMAYA-FL are treated as the same proposed method** (the tuned `jedi`
> configuration). The separate, untuned `vismaya` run appears only as a
> generative-synergy ablation.

## Compile on Overleaf
Upload **`SCOUT-FL_preliminary_report.zip`** to Overleaf (New Project → Upload Project),
then compile `main.tex` with **pdfLaTeX**. Bibliography uses BibTeX:

```
pdflatex main   →   bibtex main   →   pdflatex main   →   pdflatex main
```

`main.tex` uses `IEEEtran` (journal) + `graphicx`, `booktabs`, `amsmath`,
`xcolor[table]`, `enumitem`, `array`, `hyperref` — all standard on Overleaf.

## Contents
```
main.tex            the report (incl. a Decision-Critical Audit section; edit author line)
refs.bib            bibliography
figures/*.pdf       19 figures (13 main + figA1..A6 weight-free audit)
tables/*.tex        8 \input-ed LaTeX tables (incl. reframed publishability verdict)
decision_summary.md standalone per-method publish/reframe/reject verdict
decision_analyses.py  the audit (python decision_analyses.py -> figA*, figures/stats/*.json)
report_common.py    data loader + method taxonomy
make_figures.py     regenerates the 13 main figures
make_tables.py      regenerates tables/
SCOUT-FL_preliminary_report.zip   ← upload THIS to Overleaf
```

## Honest audit (Sec. VIII of the report + decision_summary.md)
The report's optimistic claims are corrected by a weight-free audit vs the STRONGEST
baselines: the proposed methods **trade** (not dominate), the "21/22" win is regime-specific,
and JEDI's Pareto-optimality is fragile to the CRB convention. Verdict: **publish SCOUT-FL v2
as headline, reframe JEDI, demote v1.** See `decision_summary.md` and `figA1..A6`.

## Regenerate everything from the raw runs
```
cd research/preliminary_report
python make_figures.py    # 13 figures
python make_tables.py     # 8 tables
```
Both read `runs/{campaign_main,campaign,ablation,ablation_vismaya}/**/*.json` using the
same final-round-snapshot convention as `scout_fl.analysis.collect`.

## Verdict (Sec. VIII of the report)
**Both SCOUT-FL v2 and JEDI/VISMAYA-FL are individually paper-worthy.**
- **SCOUT-FL v2** — best *balanced* method (mean joint rank 1.9/28, Nemenyi-significant);
  simple + provable. Recommended headline / co-headline.
- **JEDI/VISMAYA-FL** — best *accuracy at matched sensing*: +11.5 pp over the ISAC SOTA
  (Asaad TWC'25, p<0.01); wins 19/22 constrained-accuracy points.
- **SCOUT-FL v1** — supporting variant (dominated by v2).
- Recommendation: one unified TWC paper with both methods sharing the Pareto-frontier story.
```
```
