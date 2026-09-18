# GCLIP

Core implementation of a reusable synthetic-source classifier for rice-panicle
segmentation. The pipeline uses frozen DINOv2-S/14 features, a source-fitted GMM
and one-time source-side CLIP semantic assignment. Target inference does not
train, refit clusters, load CLIP or run a diffusion model.

## Install

Python 3.11 is the reference version. Create and activate a virtual environment,
then install dependencies and obtain the official encoder:

```bash
python -m pip install -r requirements.txt
python download_weights.py --output downloads
```

The downloader checks the official DINOv2 source and checkpoint hashes. Large
pretrained weights and third-party repositories are not included in this repository.
The source directory is `downloads/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8`.

## Segment one image or a folder

```bash
python prepare_inputs.py path/to/image_or_folder --output inputs.json
python predict.py --manifest inputs.json --dinov2-repo downloads/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8 --checkpoint downloads/dinov2_vits14_pretrain.pth --output predictions --overlay
```

Outputs: original-resolution binary masks, component maps, optional cyan overlays,
visible panicle-area fractions, input/model hashes and progress. Inference input
lists contain only `image_id` and `image`. Masks and target labels are rejected.
Use a new output directory for each run; interrupted files are preserved, not
silently reused. Duplicate image IDs and mismatched model/semantic hashes fail.

## Key implementation

| File | Role |
|---|---|
| `features.py` | 1120 letterbox, frozen 384-D DINOv2 features, float16 token round trip, dense interpolation and L2 normalization |
| `controlled_source_clustering.py`, `fit_source.py` | Out-of-core source GMM fitting and density scores |
| `extract_source.py` | Reproduce fixed source-view token sampling |
| `calibrate_clip.py` | Source-only CLIP region scoring and model-bound semantic identity |
| `calibrate_rgb.py` | RGB source-assignment alternative |
| `predict.py` | Frozen source-model transfer; weighted GMM argmax and native-size restoration |
| `metrics.py`, `evaluate.py`, `bootstrap.py`, `area.py` | Segmentation evaluation, conditional resampling and projected visible-area errors |
| `markers.py`, `count.py` | Additional validation-calibrated counting application |

The included compact NPZ is one seed42 diagonal K=3 GMM, not an ensemble.
Its component identity belongs to that exact model. New fits require new source
semantic assignment; component 2 is not universally a panicle label. There is
no PCA or segmentation postprocessing in the main inference path.

## Source construction

The 14,000-view manifest contains relative names and hashes, not images. Obtain
the fixed source archive and preserve its directory structure:

```bash
python extract_source.py --source-root path/to/synthetic_rice_200 --dinov2-repo downloads/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8 --checkpoint downloads/dinov2_vits14_pretrain.pth --output source_features
python fit_source.py --features source_features/features.npy --output source_gmm.npz --seed 42 --k 3 --covariance diag
python -m pip install -r requirements-calibration.txt
python download_weights.py --output downloads --clip
python calibrate_clip.py --source-root path/to/synthetic_rice_200 --dinov2-repo downloads/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8 --checkpoint downloads/dinov2_vits14_pretrain.pth --clip-directory downloads/clip --output semantic_assignment
```

Calibration defaults to the included density. To calibrate a new fitted density,
pass `--model` with its file/SHA256, encoder SHA256, input size and feature dimension.
The output binding is then passed to `predict.py --model`.
The complete source feature cache requires approximately 21.5 GB decimal.
No source/target images, cached features, manuscript or training logs are included.

## Evaluation and counting

```bash
python evaluate.py --manifest evaluation_ids.json --mask-root path/to/masks --predictions predictions/masks --output evaluation
python bootstrap.py --csv evaluation/per_image.csv --output intervals.json
python area.py --csv evaluation/per_image.csv --output area_errors
```

Evaluation rows contain `image_id`, `mask` and the appropriate resampling `group`.
Group labels must represent the recorded evaluation design; do not substitute
arbitrary groups. Mean image FGIoU differs from pooled mIoU/F1. An ignore encoding
requires explicit `--ignore-value`; 255 otherwise denotes foreground.

Counting is a separate application requiring real validation counts. For native
512-square images, `predict.py --probability-output` saves the compatible posterior
cache. Run `count.py --probabilities ... --manifest ... --output ...` with ID-only
test rows. `--mode calibrate` requires 537 validation records with reference counts;
the fixed protocol screens 144 rules on 90 images, then tests 24 on all 537.
Erosion-core counting uses a one-marker fallback for eligible regions without a
remaining core. It does not apply a detector, watershed or linear count correction.

## Tests and resource access

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

The core package has CPU boundary tests and real-image GPU smoke checks. A full
source refit and clean GPU installation are distinct from these software tests.
See `TEST_SCOPE.md`. Model provenance and numerical processing are documented in
`MODEL_CARD.md`, `docs/PROTOCOL.md` and `THIRD_PARTY_NOTICES.md`.

Existing source-resource share:
[Panicle_GMM_data_release](https://pan.baidu.com/s/1_3Zn60ipXP16vOrYWNqwbA), code **1111**.
Target datasets and pretrained assets remain subject to their providers' terms.
Storage availability is not a blanket license. No project open-source license
has been specified; publication of this code does not grant a license to
third-party datasets or pretrained assets. Upstream licenses remain in force.
