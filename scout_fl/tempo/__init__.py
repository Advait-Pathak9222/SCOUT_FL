"""TEMPO-FL — temporal mission planning for ISAC-FL (design Part 1).

A time-varying schedule lambda_s(t) of the learning/sensing balance fed to the
inner (SCOUT-FL) selector, chosen against explicit two-state dynamics (learning
descent + Kalman/Riccati tracking). Stationary policies (constant lambda_s) are
the degenerate special case / the null.

  schedules   — hand-crafted E-T1 schedules + naive-schedule controls (§1.5, §1.6)
  controllers — TEMPO-Threshold / -DPP / -MPC closed-loop controllers (§1.3)
  runner      — the TEMPO FL training loop (mobility + tracker + controller + selection)
"""
