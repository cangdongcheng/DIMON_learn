# Geo_MLP

Vanilla-MLP ablation for the Geo-DONet benchmark:

```text
[geometry PCA (60), Cobiveco (4), normalized time (1)] -> Tanh MLP -> V_m
```

There is no branch/trunk split and no latent dot product. This tests whether the
DeepONet factorization contributes to the error while retaining spatial and
temporal query coordinates.

A literal dense `shape(60) -> complete V_m(50797 x 121)` MLP is not used: with
width 300, its final layer alone would contain about 1.84 billion weights. The
pointwise conditional MLP is the tractable direct comparison.

## Training

Run on a GPU node:

```bash
source ~/load_dimon_env.sh
cd /home/svu/e1032484/DIMON_learn/Geo_MLP

python -u main.py \
  --device cuda \
  --epochs 5000 \
  --width 300 \
  --depth 4 \
  --case-batch-size 8 \
  --samples-per-case 4096 \
  --val-samples-per-case 16384 \
  --frame-step 5 \
  --lr-scheduled
```

Training samples node/time queries uniformly. Each heart sees 4096 newly drawn
queries whenever it appears in an epoch. Early stopping uses a larger fixed
validation sample so changes in validation loss are not caused by resampling.

The default checkpoint is:

```text
CheckPts/geomlp_w300_d4_5000ep_s4096_f121_lrsched.pt
```

If GPU memory is tight, reduce `--case-batch-size`; if training is too slow,
reduce `--samples-per-case`. Both affect the Monte-Carlo training loss, so record
them with every result.

## Test

Full fields—not sampled points—are reconstructed for all 25 held-out hearts:

```bash
python -u main.py \
  --test-model \
  --device cuda \
  --model-path CheckPts/geomlp_w300_d4_5000ep_s4096_f121_lrsched.pt
```

This reports the same full-field V_m RelL2/MAE and activation-time metrics as
Geo_DONet. It also saves `predictions.npz` and a first-case trace plot. Add
`--vtu-out` to export the first test case for ParaView.

## Fair-comparison notes

- Data: `/home/svu/e1032484/scratch/geo_donet_data_f601.npz`
- Time stride: 5, giving the same f121 grid as the benchmark
- Split: first 95 train, next 5 validation, final 25 test
- Normalization: identical train-only `Normalizer` from `Geo_DONet/utils.py`
- Activation threshold: -10 mV with linear crossing interpolation
- Architecture: same Tanh activation and default hidden width/depth

The MLP has about half as many parameters as the two-network Geo_DONet at the
same width because it contains one MLP rather than separate branch and trunk
MLPs. This is the simplest architectural ablation; parameter matching can be a
later experiment if needed.

## Five-fold cross-validation

The CV protocol matches `Geo_DONet/main_cv.py` and the run recorded in
`Geo_DONet/Geo_DONet_CV.o14298067`:

- shuffle all 125 hearts once with seed 42;
- five disjoint test folds of 25 hearts;
- for each fold, 95 fit / 5 validation / 25 test;
- refit normalization on the 95 fitting hearts only;
- select the best sampled-validation checkpoint;
- reconstruct the complete held-out fields;
- pool the five test folds so every heart is evaluated exactly once.

Submit:

```bash
cd /home/svu/e1032484/DIMON_learn/Geo_MLP
qsub cv.pbs
```

The default output directory is:

```text
CV_5fold_5000ep_w300_d4_s4096_f121_lrsched/
```

It contains a checkpoint and loss history for each fold, plus:

```text
summary.txt
per_case_metrics.csv
cv_results.npz
cv_summary.png
```

Two activation-time results are saved. `at_l2`/`at_mae` use the first 5-ms
grid frame above -10 mV, matching the historical Geo_DONet CV. The additional
`at_interp_l2`/`at_interp_mae` use linearly interpolated upward crossings,
matching the current single-split benchmark.

For a denser sampled-query follow-up, change the PBS command to:

```text
--samples-per-case 16384 --val-samples-per-case 65536
```

Use only one sampling configuration for the primary CV comparison; choosing a
configuration after examining fold test results would leak test information.
