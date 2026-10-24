# AG

Five-fold inference on a private prepared dataset.

## Setup

Use Python 3.10, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Downloads

Download the weights and prepared dataset from [Baidu Netdisk](https://pan.baidu.com/s/1EIzLG-AQYvKCt9A0lmvVlA?pwd=nprp).

Extraction code: `nprp`.

Extract both archives before running inference.

## Inputs

`--data-dir` contains the existing `manifest.json`, `splits/fold0.json` through
`splits/fold4.json`, and prepared volumes. Only the `test` patients of each
fixed fold are used; patients cannot overlap between test folds.

Each volume is an NPZ containing `ct` (float32 HU) and `mask` (binary uint8),
both shaped `(64, 64, 64)` in `(z, y, x)` order at 1 mm isotropic spacing.
Adjacent observations and their intervals in months come from the manifest.
CT normalization is applied internally.

Extract the separately supplied weights archive. `--weights` points to its
`weights/` directory, containing `manifest.json` and `model_0.pt` through
`model_4.pt`. Weight `model_i.pt` is used only for test fold `i`.

## Run

```bash
python infer_fivefold.py --data-dir /path/to/private_dataset --weights /path/to/weights --output results --device cuda:0
```

Use `--device cpu` for CPU inference. The output directory must be new or empty.
All five folds run in order; no fitting or split generation is performed.

## Outputs

Each `fold0/` through `fold4/` contains `per_pair.csv` and `metrics.json`.
`summary.json` reports the unweighted mean and sample standard deviation
across the five fold means. Within each fold, every adjacent pair has equal weight.

Metrics are DSC, TPR, PPV, PSNR, SSIM, MSE-ROI, and MSE-PN.
