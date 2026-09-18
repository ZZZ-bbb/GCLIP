# GCLIP source classifier

Research model snapshot, 2026-09-14, published with the core pipeline.
No accepted paper, universal domain invariance or open-source license is asserted.

The NPZ stores weights, means and diagonal variances of one seed42 K=3 GMM in
full 384-D DINOv2-S/14 space. `models/model.json` binds density/encoder/assignment
SHA256 values. Array loading forbids pickle; official PyTorch weights use
`weights_only=True`. Hash or semantic mismatches abort, never guess a label.

The task source is 1,400 synthetic parents and 14,000 derived views. External
DINOv2/CLIP pretraining is prior information; the main branch does not use real
task-source images/masks. CLIP assigns source semantics once and is absent at
target inference. Processing is 1120 letterbox, patch L2, float16 round trip,
float32 dense interpolation, second L2, GMM argmax, unpadding, nearest native
restoration. No PCA, target fine-tuning or postprocessing.

Retrospective image-mean FGIoU is 65.07%, 53.19%, 30.99% on RiceP85, CVRP463,
Paddy2187; per-image records define the unrounded result. Grouped intervals are
conditional, not independent flight/season confirmation. Small/green panicles
and Paddy remain difficult. Original CLIP/RGB assignment yields identical
predictions; RGB was more stable in the new recorded candidate pool. CLIP itself
is not established as an accuracy or robustness gain.

Area is a projected visible image fraction, not biomass, yield or physical area.
Main counting uses 537 real validation counts and yields MAE 12.8876, R虏 0.6682
on 1,610 test images; it is not label-free counting and is not historical Real/Dual.
Do not use this classifier alone for breeding, pesticide or harvest decisions.
Validate organ identity/scale/coverage before deployment. Negative-image
specificity and untouched seasons are not established by these positive-image tests.
