# snn-surrogate-baseline

Surrogate-gradient backprop baseline for spiking neural networks, following
[snnTorch Tutorial 6](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)
on MNIST. This is the **backprop reference** the Neuron Games proposal's three-factor
learning rule and Eligibility-Shapley credit assignment get measured against.

Scope: reproduce the tutorial faithfully, on our hardware, with the reproducibility
and measurement scaffolding a real experiment needs. Explicitly NOT the full
Neuron Games framework — that's William's STDP work and the eventual game
formulation, both landing separately.

## Hardware / environment on this machine

- GPU: **NVIDIA GeForce RTX 5080** (Blackwell, `sm_120`), 16 GB VRAM
- Driver: 610.88 · CUDA UMD 13.3
- OS: Windows 11 (10.0.26100)
- Python: 3.12.14 (uv-managed)
- PyTorch: **2.11.0+cu128** · torchvision 0.26.0+cu128 · cuDNN 91900
- snnTorch: **1.0.0**
- `nvidia-ml-py`: 13.610.43

Blackwell only works on cu128+ wheels; older CUDA builds either fail with
"no kernel image" or slow-path silently.

## Setup (from scratch)

Install [uv](https://github.com/astral-sh/uv), then from this folder:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0" "torchvision==0.26.0"
uv pip install --python .venv/Scripts/python.exe \
    "snntorch==1.0.0" "matplotlib==3.11.2" "nvidia-ml-py==13.610.43" \
    jupyterlab ipykernel
```

`requirements.lock.txt` has the full resolved set (transitive deps and hashes).

Verify GPU (fails loudly if broken — never falls back to CPU):

```bash
.venv/Scripts/python.exe scripts/gpu_check.py
```

## Commands

All commands run from `snn-surrogate-baseline/` with the venv Python.

| Purpose | Command |
| --- | --- |
| Smoke test (~10 s) | `.venv/Scripts/python.exe -m neuron_baseline smoke` |
| One-epoch baseline (default cfg) | `.venv/Scripts/python.exe -m neuron_baseline train` |
| Longer run w/ overrides | `.venv/Scripts/python.exe -m neuron_baseline train --epochs 5 --seed 42` |
| Evaluate a saved checkpoint on the test set | `.venv/Scripts/python.exe -m neuron_baseline eval --ckpt runs/<TS>/checkpoints/best.pt` |
| Notebook end-to-end run (same code path as the CLI) | `.venv/Scripts/jupyter lab notebooks/baseline_walkthrough.ipynb` |

Each `train` invocation writes to `runs/<timestamp>/` with:

```
runs/<TS>/
├── config.json                 # resolved config + software/hardware fingerprint
├── results.json                # test acc, val history, wall time, energy JSON
├── checkpoints/
│   ├── best.pt                 # selected on best validation accuracy
│   └── last.pt                 # end-of-training snapshot
└── plots/
    ├── loss.png
    └── accuracy.png
```

## CIFAR-10 / CIFAR-100

The same baseline also runs on CIFAR-10 and CIFAR-100. MNIST stays the default
and its code path is unchanged.

| Purpose | Command |
| --- | --- |
| CIFAR-10 on the GPU | `.venv/Scripts/python.exe -m neuron_baseline train --dataset CIFAR10 --epochs 20 --num-workers 4` |
| CIFAR-100 on the GPU | `.venv/Scripts/python.exe -m neuron_baseline train --dataset CIFAR100 --epochs 20 --num-workers 4` |
| Without augmentation | add `--no-augment` |
| Input pipeline walkthrough | `notebooks/cifar_input_pipeline.ipynb` |

On Linux / macOS / Colab use `python -m neuron_baseline ...` instead of
`.venv/Scripts/python.exe`. On a Mac without CUDA add `--device mps`
(no NVML energy numbers there).

What changes for CIFAR (`neuron_baseline/data.py`, `neuron_baseline/model.py`):

- **Input:** RGB 32×32, so the first conv takes 3 channels and the flattened
  size becomes 64·5·5 = 1600 (MNIST: 64·4·4 = 1024).
- **Output:** 10 LIF neurons for CIFAR-10, 100 for CIFAR-100.
- **Normalization:** per-channel mean / std of the training set
  (CIFAR-10 `(0.4914, 0.4822, 0.4465)` / `(0.2470, 0.2435, 0.2616)`,
  CIFAR-100 `(0.5071, 0.4865, 0.4409)` / `(0.2673, 0.2564, 0.2762)`).
- **Augmentation (train split only):** `RandomCrop(32, padding=4)` +
  `RandomHorizontalFlip`. The validation split is carved from the training set
  with the same seeded permutation but always uses the clean transform.
- **Split:** 45,000 train / 5,000 val / 10,000 test, same seed and split logic as MNIST.
- **Encoding:** unchanged — the normalized image is presented at each of the T steps.
- Layer order is identical, so `diversity_layer_idx=5` still taps the second Leaky layer.
- Run folders are named `runs/<dataset>-<timestamp>/`; `results.json` records
  `dataset` and `augment`.

Notes: the Tutorial 6 hyper-parameters (lr = 1e-2, T = 50, 1 epoch) were tuned
for MNIST. On CIFAR the network learns slowly with them, so use more epochs
(`--epochs 20`) and compare `--lr 1e-3`. This small conv SNN is a reference
point for the comparison, not a state-of-the-art CIFAR model.

## Model — Tutorial 6 verbatim

```
Conv2d(1, 12, 5) → MaxPool2d(2) → Leaky
Conv2d(12, 64, 5) → MaxPool2d(2) → Leaky
Flatten() → Linear(1024, 10) → Leaky(output=True)
```

Every `snn.Leaky`:
- `beta = 0.5`
- `spike_grad = fast_sigmoid(slope=25)`
- `threshold = 1.0` (snnTorch default)
- `reset_mechanism = 'subtract'` (snnTorch default: `u ← u − threshold` on spike, not zero-reset)
- `init_hidden = True` (so `nn.Sequential` composes and hidden state is stored on the module)

Only the last also sets `output=True` so it returns `(spk, mem)`.

## Training loop

- MNIST, batch size 128, T = 50 simulation steps.
- Same pixel values presented at each of the T steps (Tutorial 6's "direct"
  input path). No Bernoulli / latency encoding.
- Adam, lr = 1e-2, betas = (0.9, 0.999).
- Loss: `SF.ce_rate_loss()`.
- Accuracy: `SF.accuracy_rate()`.
- Between batches: `utils.reset(net)` restores neuron hidden state.
- Within a batch: hidden state carries across the T loop by design.
- 1 epoch by default; longer runs configurable via `--epochs`.

## Facts pinned from the installed snnTorch (1.0.0), not just the tutorial prose

Inspected in-repo before writing this so we don't rely on drifting docs:

- **`SF.ce_rate_loss` temporal reduction is a *mean* over T, not a sum.** Source:
  `for step in range(num_steps): loss += loss_fn(log_p_y[step], targets); return loss / num_steps`.
  The tutorial text says "accumulated" — factually it's `accumulated / T`. This is what
  our loss curves plot.
- **`SF.accuracy_rate` sums spikes over T, argmax on the class dim.** It is not
  a per-step majority vote. Our `evaluate()` uses the same rule so training and
  validation accuracies are computed identically.
- **Reset gradient is detached.** `SpikingNeuron.mem_reset()` returns
  `spike_grad(mem - threshold).clone().detach()`. So gradient does NOT flow
  *through* the subtract-reset term; only the surrogate-gradient path
  through the spike function carries gradient. This is a snnTorch design
  choice, not something the tutorial documents.
- **`utils.reset(net)` walks every `snn.SpikingNeuron` submodule and calls
  `.reset_hidden()`**. Only affects neurons with `init_hidden=True`, which is
  all of ours.

## Our experimental research adaptations (labelled as ours, not the tutorial)

The tutorial uses one train loader and one test loader, both with
`drop_last=True`, and no validation set. Deviations:

1. **Fixed validation split** of 5000 examples carved off the front of the
   shuffled (seeded) MNIST training set. Split is seeded independently of the
   dataloader RNG so shifting the global seed doesn't quietly reshuffle.
2. **Checkpoint selection on validation accuracy**. `best.pt` is the epoch
   with highest val acc; `last.pt` is end-of-training.
3. **Test set is reserved for final evaluation only** — used exactly once per
   run, on the loaded `best.pt`.
4. **Every example evaluated**, including incomplete final batches. Eval
   loaders use `drop_last=False`. Loss is size-weighted so per-batch means
   compose correctly when the last batch is smaller.
5. **Eval batch size decoupled from train batch size** (default 256 for eval,
   128 for train).
6. **Resolved config + full software/hardware fingerprint saved with every run.**

## Reproducibility limits

Seeds cover Python `random`, NumPy, and PyTorch (`manual_seed` + `cuda.manual_seed_all`).
We **do not** flip cuDNN into deterministic mode — the benchmarked kernels are
noticeably faster and we prefer the throughput on this hardware. Consequence:
convolution kernel selection and reduction ordering vary between runs, so
final test accuracy may differ by a fraction of a percent even with identical
seed and config. If bit-exact reproduction becomes important later, set
`torch.backends.cudnn.deterministic = True` and `benchmark = False` in
`neuron_baseline/reproducibility.py`.

## Energy measurement — what it actually reports

`neuron_baseline/energy.py` uses NVML with graceful fallback:

1. **Preferred:** `nvmlDeviceGetTotalEnergyConsumption` — a monotonic mJ
   counter. Read at start and end of a window, difference is exact energy
   over that window.
2. **Fallback:** `nvmlDeviceGetPowerUsage` (mW) sampled on a background
   thread at 20 Hz. Trapezoidal integration:

   > E ≈ Σ [(P_k + P_{k+1}) / 2] · (t_{k+1} − t_k)

   Energy in joules, power in watts, time in seconds.
3. **Unavailable:** if NVML can't be initialized, the report shows
   `energy_j = null` and `method = "unavailable"`. **We do not estimate
   joules from runtime or spike counts.**

Every measurement window synchronizes CUDA at both boundaries so deferred
kernels don't leak between windows.

**Recorded per window:**

- **Scope:** **GPU-only** device reading (RTX 5080). Does NOT include CPU,
  system RAM, PSU efficiency, or any host-side overhead. Do not label this
  whole-system energy or an official MLPerf figure.
- Method used (`cumulative_energy` / `power_sampling` / `unavailable`).
- Sample interval (fallback path) — 20 Hz default; results are still noisy
  at this rate.
- Warm-up before the inference-window measurement: `warmup_batches` (default 2)
  forward-only passes on the val loader.
- Batch size and precision (float32; no AMP).
- Operations included: `train` window covers the full training loop
  including per-epoch validation eval (they are not separated in a
  one-epoch run — noted in `results.json`). `inference` window covers the
  final test-set pass on the reloaded `best.pt`.
- Any concurrent GPU workloads on the desktop (ChatGPT/Slack/Teams GUIs each
  hold a handle) show up in the reading — flagged in the notes field.

If reporting an idle-adjusted number later, do it separately with the
subtraction method explained and the raw GPU energy still reported as the
primary figure.

## Layout

```
snn-surrogate-baseline/
├── README.md
├── requirements.txt / requirements.lock.txt
├── neuron_baseline/
│   ├── __init__.py
│   ├── __main__.py            # `python -m neuron_baseline {smoke,train,eval}`
│   ├── config.py              # Config dataclass — every knob + fingerprint
│   ├── data.py                # MNIST / CIFAR-10 / CIFAR-100 loaders + fixed val split
│   ├── model.py               # tutorial architecture + forward_pass(T)
│   ├── train.py               # training loop + checkpoint save
│   ├── eval.py                # full-set evaluation, variable batch size
│   ├── runner.py              # end-to-end runner shared by CLI and notebook
│   ├── plots.py               # loss/accuracy figures
│   ├── energy.py              # NVML cumulative + power-sampling fallback
│   └── reproducibility.py     # seeding + software/hardware fingerprint
├── tests/
│   └── smoke.py               # shape / grad / reset / checkpoint round-trip
├── scripts/
│   └── gpu_check.py           # standalone GPU sanity check
├── notebooks/
│   ├── baseline_walkthrough.ipynb   # calls runner.run_training(cfg) — full path
│   └── cifar_input_pipeline.ipynb   # CIFAR data input pipeline walkthrough
├── data/                      # MNIST (git-ignored)
└── runs/                      # per-run outputs (git-ignored)
```
