# Quality Control for Medical Image Segmentation Under Domain Shift With Heteroscedastic Regression

Quality control (QC) experiments for segmentation reliability. This repo trains UNet models, trains score predictors for QC, and evaluates multiple QC baselines (score-agreement, Mahalanobis, and predictive-entropy) with a shared analysis notebook.

## Project layout
- [src/apps](src/apps) — training and evaluation entrypoints.
- [src/notebooks/QC_eval.ipynb](src/notebooks/QC_eval.ipynb) — QC analysis (ranking and risk-control).
- [src/notebooks/unet_eval_auto_vis.ipynb](src/notebooks/unet_eval_auto_vis.ipynb) — UNet eval aggregation and visualization notebook.
- [results/](results/) — saved outputs (per-dataset/split/method runs).
- [pre-trained/](pre-trained/) — pretrained checkpoints.

## Installation
TODO

## Training
UNet training:
- [src/apps/train_unet_per_split.sh](src/apps/train_unet_per_split.sh)

Score predictor training (Beta$_{\mu,\kappa}$ QC head on top of UNet):
- [src/apps/train_score_predictor.sh](src/apps/train_score_predictor.sh)
- [src/apps/train_score_predictor.py](src/apps/train_score_predictor.py)

## Evaluation baselines
These scripts compute QC signals and save them into results files for later aggregation:
- Score agreement (SA): [src/apps/eval_score_agreement.sh](src/apps/eval_score_agreement.sh)
- Mahalanobis distance (Maha): [src/apps/eval_mahalanobis.sh](src/apps/eval_mahalanobis.sh)
- Predictive entropy (PE): [src/apps/eval_comp_entropy.sh](src/apps/eval_comp_entropy.sh)

## Analysis notebooks
QC analysis workflow is in [src/notebooks/QC_eval.ipynb](src/notebooks/QC_eval.ipynb). It:
- Loads results for multiple datasets/splits and all runs.
- Fits calibrators (thresholding for correlation-based methods; beta adapters for beta-based predictors).
- Computes ranking metrics (Pearson’s $\rho$, MAE, eAURC) and risk-control metrics (Rec<sup>+</sup> / Rec<sup>-</sup> at t=0.8, α=0.95).

UNet evaluation aggregation and visualization is in [src/notebooks/unet_eval.ipynb](src/notebooks/unet_eval.ipynb).

## Notes
- Results are organized by dataset, split, method, and run ID under [results/](results/).
- Dataset shifts used in the paper:
	- M\&Ms scanner drift → `scanner-symphonytim`
	- M\&Ms pathology drift → `pathology-norm-vs-fall-scanners-all`
	- PMRI dataset shift → `promise12`
	- PMRI 3T→1.5T shift → `threet-to-onepointfivet`