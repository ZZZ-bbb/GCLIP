# Reproducibility scope

1. Fixed-model inference: exact source GMM file, semantic binding, encoder hash,
   1120 letterbox RGB, ImageNet normalization, final 384-dimensional S/14 patch
   features, L2 normalization, float16 cache round trip, float32 bilinear dense
   interpolation, second L2 normalization, mixture-weighted component argmax,
   padding removal and nearest-neighbor native label restoration. No PCA.
2. Density refitting: identical view/token manifests and feature sampling are
   needed in addition to fitting code and seed. Source views are correlated,
   not independent biological replicates. `fit_source.py` expects NumPy shape
   (views, 2000, 384); fitting is not needed for the released classifier.
3. Exact synthetic-image regeneration: not established from the surviving prompt
   record alone. Generator version, sampling configuration and licensing must be
   completed by the authors. Published fixed image bytes and hashes can instead
   define a reproducible input resource.

The source-only semantic stage uses OpenAI CLIP ViT-B/32; three exact prompts
per class; unit text embeddings averaged and renormalized; full-frame component
pixels preserved against RGB (123,117,104); official 224 bicubic center crop;
cosine(panicle) minus max(cosine(leaf), cosine(background)); equal-image mean;
largest mean with lower-index ties. Empty components abort the calibration set.
Inference does not call CLIP after the component identity has been frozen.

Count markers are not instance-boundary annotations. Erosion-core counting uses
8-connected qualified parents, 4-neighbor erosion, eligible 8-connected cores,
one probability-maximum marker per core, and one marker for a qualified parent
with no eligible core. It applies no watershed or linear bias correction.
