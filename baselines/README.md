# baselines

Reference models for the Neuron Games comparison. Backprop baselines
(CNN, ResNet, UNet, Transformer) share one training loop so their numbers
are comparable; non-backprop baselines (Hopfield, ART, SOM) come later.

Data split is shared with `snn-surrogate-baseline/` (see `common/data.py`).

## Layout

| Path | Purpose |
| --- | --- |
| `run.py` | Entry point: pick model, seed, config |
| `common/data.py` | MNIST, CIFAR-10 and CIFAR-100 loaders (MNIST path copied from `snn-surrogate-baseline`) |
| `common/train.py` | Shared train/eval loop for backprop models |
| `common/nonbp.py` | Shared fit/eval helpers for non-backprop models |
| `models/` | One file per architecture |
| `prefetch_data.py` | Download all 3 datasets once (run where there is internet) |
| `smoke_test.sh` | Quick check that all four models run (do this first) |
| `run_all_baselines.sh` | Full offline job: 4 models × 3 datasets × seeds |
| `compile_results.py` | Compile every `results.json` into one `BASELINE_RESULTS.md` |
| `results/` | One folder per run (gitignored) |

## Offline / PACE workflow (TL;DR)

```
# once, where there IS internet (e.g. a PACE login node):
python prefetch_data.py --data-dir data

# then on the GPU node (offline is fine):
bash smoke_test.sh                 # ~2 min: confirms all four work
SEEDS="1 2 3" bash run_all_baselines.sh    # full job, writes BASELINE_RESULTS.md
```

## Setup

```
cd baselines
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```
python run.py --model cnn                                  # MNIST (default)
python run.py --model cnn --dataset cifar10
python run.py --model cnn --dataset cifar100
python run.py --model cnn --dataset cifar10 --no-augment   # disable crop/flip augmentation
python run.py --model transformer                          # ViT on MNIST
python run.py --model transformer --dataset cifar10
python run.py --model transformer --dataset cifar100 --dim 192 --depth 6 --heads 6
./run_all.sh transformer                                   # all three datasets, one after the other
```

`--model` picks the architecture (`cnn`, `transformer`; `python run.py --help`
lists the model-specific flags). `--dataset` picks MNIST (1x28x28) or
CIFAR-10/100 (3x32x32). Every model adapts its input channels, feature size and
number of classes from the dataset; with MNIST the CNN is the same model as
before (50,378 parameters).

`--lr` defaults per model (CNN 1e-3, Transformer 3e-4) and is recorded in
`config.json` / `results.json`. `--download` / `--no-download` controls dataset
fetching (use `--no-download` offline after `prefetch_data.py`).

**Transformer** (`models/transformer.py`) is a small Vision Transformer:
`patch_size` x `patch_size` patches -> linear embedding (`dim`) + [CLS] token +
learned positional embedding -> `depth` pre-norm encoder blocks
(`heads`-head self-attention, MLP of width `dim * mlp_ratio`) -> [CLS] head.
Defaults `--patch-size 4 --dim 192 --depth 6 --heads 6 --mlp-ratio 2 --dropout 0.1`
give ~1.8M parameters (49 patches on MNIST, 64 on CIFAR) — enlarged from the
first version (~540k) so it has the capacity for CIFAR-100.

### Non-backprop baselines (Hebbian / competitive, no gradients)

These do **not** use `common/train.py`; each has its own `fit`/`predict` and is
run through the same `run.py`. Augmentation is forced off (they learn from raw
pixels) and `--lr` is ignored except by SOM.

```
python run.py --model hopfield --dataset mnist
python run.py --model hopfield --dataset cifar100 --protos-per-class 16 --mode continuous
python run.py --model art --dataset cifar10 --vigilance 0.8 --epochs 3
python run.py --model som --dataset cifar100 --grid 30 --som-lr 0.5 --epochs 10
```

- **hopfield** (`models/hopfield.py`) — associative-memory classifier. Now stores
  **`--protos-per-class` prototypes per class** (k-means within each class, default
  8), not a single template, so it can represent several sub-clusters per class —
  the main fix for multi-class/CIFAR. `--mode bipolar` is classic Hopfield (±1
  patterns, overlap = negative energy); `--mode continuous` keeps real-valued
  prototypes and scores by cosine (no binarization loss). `--protos-per-class 1
  --mode bipolar` reproduces the original single-template model. One-shot, no
  epochs.
- **art** (`models/art.py`) — simplified Fuzzy **ARTMAP** (supervised). Online
  category creation with a vigilance test (`--vigilance`); each category carries a
  class label (match-tracking on clashes). Records category growth per epoch and
  warns if `--max-categories` is hit. Inputs are per-image min-max normalized
  (documented in the module). Test-time read-out = label of the highest-choice
  category (standard ARTMAP; no vigilance reset at test, by design).
- **som** (`models/som.py`) — Kohonen **self-organizing map**, a `--grid` x `--grid`
  sheet of prototypes trained by competitive updates with linearly decaying rate
  (`--som-lr`) and neighbourhood radius (documented schedule). Cells labelled by
  training-set majority vote; empty/unlabelled cells are tracked and reported. Now
  uses **batched updates** (≈100× faster than per-sample) and keeps the
  best-validation map.

All three now **keep and restore the best-validation checkpoint** before the test
pass (same convention as the backprop models), and record diagnostics
(`results.json → metrics`: prototype/category/empty-cell counts, `n_train_used`).

> ART is O(samples × categories) and slow on full data (hours). `run_all_baselines.sh`
> caps it at `ART_SUBSET=15000` by default; override with `ART_SUBSET=0` for all
> data. SOM is now fast (batched) and uses all data by default.

### Adding a model

Backprop (e.g. Richard's ResNet): `models/<name>.py` with an `nn.Module`
constructor `Model(num_classes, in_channels, image_size, **extra)`, register in
`BP_MODELS`. Non-backprop: same constructor but implement
`fit(train_loader, val_loader, epochs, device, run_dir)` + `predict(x)` instead
of `nn.Module`, and register in `NONBP_MODELS`. Extra CLI flags go in `MODEL_ARGS`.

CIFAR uses per-channel mean/std normalization and, by default, random crop
(padding 4) + horizontal flip on the training split only. The 5,000-image
validation split is taken from the training set with the same seed and is never
augmented. Each run writes `results/<dataset>/<model>/<run>/` with
`config.json`, `results.json`, `best.pt` and `curves.png`.

CNN reference results (seed 20260923, 25 epochs, T4 GPU):

| Model | Dataset | Augmentation | Params | Test acc |
| --- | --- | --- | --- | --- |
| CNN | MNIST | - | 50,378 | 99.14% |
| CNN | CIFAR-10 | yes | 60,554 | 76.48% |
| CNN | CIFAR-10 | no | 60,554 | 73.12% |
| CNN | CIFAR-100 | yes | 429,284 | 49.70% |

Transformer (now ~1.8M params) and the non-backprop baselines are produced by
`run_all_baselines.sh` and compiled into `BASELINE_RESULTS.md` — re-run after
these upgrades (the enlarged Transformer and multi-prototype Hopfield change the
earlier numbers). `compile_results.py` picks the newest valid run per
model/dataset and ignores failed folders; pass `--average` for multi-seed
mean ± std.
