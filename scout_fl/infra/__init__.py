"""Phase-0 shared infrastructure for the TEMPO-FL / CloakFL program
(research/RESEARCH_DESIGN_TEMPO_CLOAKFL.md §0.2):

  mobility  — constant-velocity Gaussian random-walk target motion + trajectory log
  tracker   — per-target information-filter Kalman tracker (+ NEES consistency)
  leakage   — client-position Fisher-information leakage accountant (CloakFL metric)
  dither    — M2 zero-sum correlated dithering + eavesdropper CRB model
  replay    — deterministic (config, seed) -> scenario reconstruction for re-scoring
              existing runs/ artifacts (E-C4, static-frontier tracker re-scoring)
"""
