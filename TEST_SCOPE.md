# Core-package test scope

- Python 3.11; CPU protocol tests cover empty/full foreground, ignore support,
  non-square geometry, semantic identity, tie handling, component unions, RGB
  coefficients, source sampling and erosion fallback.
- The preceding full candidate passed 25 tests. This reduced core repository
  retains the 22 method tests and omits publication-tool tests with that tool.
- Three real target images (first fixed ID per dataset) were run with the same
  hash-bound encoder/GMM in the pinned existing CUDA research environment.
  The portable full candidate exactly reproduced archived confusion counts and
  native masks from the earlier smoke run.
- Original 20-source-image CLIP and RGB recalibration selected component 2.
  Two actual source views were extracted and a two-iteration EM software smoke
  completed; this is not a converged full-source refit.
- No remote CI, remote-clone reproduction or fresh GPU installation is claimed
  until such checks are actually performed.
