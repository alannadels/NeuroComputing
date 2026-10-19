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
| `models/` | One file per architecture |
| `results/` | One folder per run (gitignored) |

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
```

`--dataset` picks MNIST (1x28x28) or CIFAR-10/100 (3x32x32). The CNN adapts its
input channels, flattened feature size and number of classes from the dataset;
with MNIST it is the same model as before (50,378 parameters).

CIFAR uses per-channel mean/std normalization and, by default, random crop
(padding 4) + horizontal flip on the training split only. The 5,000-image
validation split is taken from the training set with the same seed and is never
augmented. Each run writes `results/<dataset>/<model>/<run>/` with
`config.json`, `results.json`, `best.pt` and `curves.png`.

Reference results (seed 20260923, 25 epochs, T4 GPU):

| Dataset | Augmentation | Params | Test acc |
| --- | --- | --- | --- |
| MNIST | - | 50,378 | 99.14% |
| CIFAR-10 | yes | 60,554 | 76.48% |
| CIFAR-10 | no | 60,554 | 73.12% |
| CIFAR-100 | yes | 429,284 | 49.70% |
