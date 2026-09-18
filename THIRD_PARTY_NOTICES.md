# Provenance and third-party terms

- DINOv2: [Facebook Research official source](https://github.com/facebookresearch/dinov2),
  Apache-2.0 source license. The downloader retains LICENSE. All registered research
  Python files/hubconf.py match revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`;
  hashes are in `configs/dinov2_source.json`. The historical Torch Hub cache did
  not retain a git commit: this records verified source parity, not a recovered log.
- CLIP: [OpenAI official code](https://github.com/openai/CLIP) and official
  `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
  Downloads use recorded SHA256 values. Upstream model/code terms apply; no large
  weights are vendored or relicensed.
- NumPy, SciPy, OpenCV, Pillow, PyTorch, torchvision, scikit-learn, timm and
  transformers remain separately licensed installed dependencies.
- Controlled clustering/fitting derive from the preceding GCLIP research export;
  features/semantic routines from its semantic_assignment.py; metrics from the
  grouped-study evaluator; marker functions from the recorded DRPD evaluation
  script. This is a portability/test refactor, not a new invention claim.
  `docs/PROVENANCE.json` records source identities.
- Baseline implementations/weights are not copied under a new license.
  Diagnostic tables describe transferred conditions; CAUSE is not certified as
  a successful original-benchmark reproduction.

No blanket project license is inferred from storage visibility. Code, GMM,
prompts, synthetic/real images and third-party figures have separate rights.
Public redistribution requires corresponding authorization.
