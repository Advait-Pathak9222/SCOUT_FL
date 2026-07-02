"""CloakFL — location-private client participation in ISAC-FL (design Part 2).

  entanglement — E-C1 target/leakage FIM frontier + GATE 2 (analytic kill test)
  dither_study — E-C2 zero-sum dither validation + GATE 3 (analytic + short FL)
  mechanisms   — M2 dither / M3 obfuscation / naive privacy mechanisms (DP-Gaussian,
                 coarse-grid, SecAgg-only, uniform-power)
  selection    — M1 leakage-capped selection + privacy-baseline selection variants
  runner       — CloakFL FL training loop (E-C3 privacy-utility frontier, E-C5 rotation)
"""
