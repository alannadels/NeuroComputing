# NeuroComputing: Neuron Games

Research repository for the **Mukhopadhyay CDDM** effort: training spiking
neural networks **without backpropagation**.

## The project

Backpropagation is the standard way to train neural networks, but it needs a
backward pass that carries exact error gradients through every layer and,
for spiking or recurrent networks, back through every time step (BPTT). That
costs memory proportional to the sequence length and is widely considered
biologically implausible.

This project implements and evaluates the proposal

> **"Neuron Games for Spiking Networks: A Constrained State-Based Potential
> Markov Game with Kernel/DPP Diversity, Metabolic Barriers, and
> Eligibility-Shapley Explainability"**
> (`neuron-games/docs/papers/mukhopadhyay-cddm.pdf`)

The goal is a learning rule that **matches or approaches backpropagation's
accuracy while using less training compute**, measured head to head against a
surrogate-gradient BPTT network of the same architecture.

### The approach in brief

- **Neurons are players.** Each neuron's action is the change to its incoming
  synapses. All neurons share one potential
  `Phi = -J + beta*V - mu*E - lambda*R`: task loss `J`, within-layer diversity
  `V`, metabolic and barrier energy `E`, weight regularizer `R`.
- **Three-factor plasticity is the learning rule.** Each synapse keeps a local
  eligibility trace; a broadcast learning signal (the third factor) decides how
  much of it to apply: `dW_ij = -eta * sum_t m_i[t] * e_ij[t]`.
- **No backward pass.** The output error reaches hidden neurons through fixed
  random feedback weights (e-prop style). Every term of `Phi` contributes a
  per-neuron signal that the eligibility trace turns into a weight change.
- **Diversity and homeostasis keep the game stable**: diversity stops neurons
  from all learning the same feature; energy costs and barriers stop them from
  firing at maximum rate to capture credit.
- **Eligibility-Shapley credit** gives each neuron a tractable share of the
  improvement in `Phi`, used both to scale its learning signal during training
  and to explain the trained network afterwards.

To reproduce the experiments, see **[Running the full pipeline](#running-the-full-pipeline)**.
The original architecture and work split are in
[`Architecture_Work_14-10-2026.md`](Architecture_Work_14-10-2026.md).

## Repository structure

```
NeuroComputing/
├── README.md                          this file: project, how to run, design decisions
├── Architecture_Work_14-10-2026.md    original work split (all parts now implemented)
├── LICENSE
├── neuron-games/                      the Neuron Games method (main project)
│   ├── README.md                      proposal summary, commands, open questions
│   ├── requirements.txt               pinned environment
│   ├── docs/
│   │   ├── papers/                    the proposal and reference papers
│   │   ├── experiments/               tuning reports and results tables (.md, .tex)
│   │   └── implementation-plan.md     original implementation plan
│   ├── src/neuron_games/
│   │   ├── config.py                  conditions (ablation rows) and all hyperparameters
│   │   ├── model.py                   784-128-10 LIF network; records eligibility traces
│   │   ├── encoding.py, data.py       spike encoding; MNIST / Fashion-MNIST loaders
│   │   ├── learning_rule.py           three-factor rule
│   │   ├── game.py, state.py          one adaptation window (proposal section 11), game state
│   │   ├── guard.py                   no-backprop enforcement and autograd-call audit
│   │   ├── train.py                   training loop, LR schedule, per-epoch metrics
│   │   ├── eval.py, test_metrics.py   evaluation and the detailed test metrics
│   │   ├── timing.py, progress.py     GPU-correct timing, peak memory, progress bar
│   │   ├── runner.py                  one full run -> results.json
│   │   ├── potential/                 terms of Phi: task J, diversity V, energy E, regularizer R
│   │   ├── modulator/                 random feedback broadcast, Eligibility-Shapley credit
│   │   └── explain/                   faithfulness, stability, sparsity of credit
│   ├── scripts/
│   │   ├── run_snn.py                 train + test one model, or evaluate a checkpoint
│   │   ├── run_comparison.sh          full head-to-head pipeline (no backprop vs backprop)
│   │   ├── compare_report.py          comparison.md / .json and plots from a run directory
│   │   ├── latex_tables.py            LaTeX tables from comparison.json
│   │   ├── run_ablation.sh, aggregate.py, beta_sweep.py   ablation grid and sweeps
│   ├── tests/                         pytest suite (includes the no-backprop checks)
│   └── data/, runs/                   datasets and run outputs (both git-ignored)
└── snn-surrogate-baseline/            snnTorch Tutorial 6 surrogate-gradient BPTT baseline
                                       (conv SNN); neuron-games imports its data loaders,
                                       seeding, energy meter, plots and kernel library
```

## Running the full pipeline

This reproduces the head-to-head comparison of the no-backprop SNN (`ng_snn`,
full Neuron Games) against the backprop SNN (`bp_snn`, surrogate-gradient
BPTT) on the same 784-128-10 network, data split, encoding, loss and seeds,
with the full metric suite and a report.

### 1. Requirements

- Python 3.12 and an NVIDIA GPU with CUDA 12.8 support. Any recent NVIDIA GPU
  works; the reported results were obtained on a single NVIDIA RTX 5090. CPU and
  Apple Silicon also work for small runs.
- About 1 GB free disk for MNIST and run outputs.
- Internet access on the first run (MNIST downloads automatically to
  `neuron-games/data/raw/`).

### 2. Get the code and set up the environment

```bash
git clone git@github.com:alannadels/NeuroComputing.git
cd NeuroComputing/neuron-games

# with uv (https://github.com/astral-sh/uv)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
uv pip install -r requirements.txt
```

With plain pip, replace `uv pip` by `pip` after creating the virtualenv with
`python3.12 -m venv .venv`. On CPU or Apple Silicon, install `torch` and
`torchvision` from the default index (drop `--index-url`).

### 3. Check the installation

```bash
python -m pytest tests -q
```

All tests should pass (two snnTorch cross-checks skip if `snntorch` is not
installed). The suite includes the no-backprop checks: every learning signal
is compared against autograd, and training under every no-backprop condition is
run with an autograd-call counter that must stay at zero.

### 4. Run the comparison

Every command below is run from `neuron-games/`.

```bash
./scripts/run_comparison.sh                                  # 1 seed, 50 epochs, MNIST
./scripts/run_comparison.sh --seeds "1 2 3"                  # the 3-seed comparison we report
```

The script:

1. runs the test suite and checks the GPU (aborts before training if either fails);
2. records the git commit, GPU model, driver and package versions in `<out>/env/`;
3. for each seed, trains `bp_snn` then `ng_snn` for 50 epochs and evaluates the
   best-validation checkpoint once on the test set; with several seeds, the
   first seed finishes for both models first and gets an interim report;
4. runs the explainability report on every run;
5. writes the head-to-head report `<out>/comparison.md` (plus
   `comparison.json` and plots).

It exits non-zero if any run fails, and the report fails if the no-backprop
model made a single autograd call during training. Expect about 6.5 minutes
per model per seed on the RTX 5090 the results were obtained on (about 40
minutes for 3 seeds); slower GPUs take proportionally longer.

Options: `--dataset mnist|fashion_mnist`, `--epochs N`, `--seeds "..."`,
`--models "bp_snn ng_snn"` (any two of `bp_snn`, `ng_snn`, `three_factor`,
`tf_div`, `tf_div_energy`, `ng_full_st2`), `--device cuda|mps|cpu`,
`--out <dir>`, `--first-seed-only`, `--skip-tests`, `--skip-explain`. Anything
after `--` goes to every training run, e.g.
`./scripts/run_comparison.sh -- --set beta=1e-3 --set lr_patience=3`.

On a shared GPU machine, run it inside `tmux` (or a job queue) so it survives
a disconnect:

```bash
tmux new -s compare
./scripts/run_comparison.sh --seeds "1 2 3"
# detach: Ctrl-b then d      reattach: tmux attach -t compare
```

Seeds run in separate invocations can be combined into one report, as long as
they used the same commit and settings (check `env/run_info.txt`):

```bash
mkdir -p runs/combined/bp_snn runs/combined/ng_snn
cp -r runs/<first-run>/bp_snn/seed-1 runs/<second-run>/bp_snn/seed-* runs/combined/bp_snn/
cp -r runs/<first-run>/ng_snn/seed-1 runs/<second-run>/ng_snn/seed-* runs/combined/ng_snn/
python scripts/compare_report.py runs/combined
```

### 5. Outputs

```
runs/compare-<dataset>-<timestamp>/
├── comparison.md / comparison.json      head-to-head report (mean ± std over seeds)
├── val_acc_vs_epoch.png, val_acc_vs_time.png, acc_vs_step.png, steps_sweep.png
├── results.jsonl                        one summary line per run
├── run_comparison.log                   console log
├── env/                                 commit, GPU, packages, test results
└── <model>/seed-<s>/
    ├── results.json                     every metric of the run (see below)
    ├── config.json                      resolved configuration
    ├── explain.json, plots/             explainability report and training plots
    ├── train.log                        that run's console output
    └── checkpoints/{best,last}.pt
```

Metrics, identical for every model:

- **Training (per batch / per epoch):** loss, accuracy, firing rates, the
  model's own potential Phi and reward, gradient and update norms,
  Eligibility-Shapley credit per neuron, a reference Phi that scores every model
  with the same fixed terms, learning rate, training and validation time, peak
  GPU memory for training and validation, GPU energy, autograd-call count.
- **Test:** accuracy, balanced accuracy, macro precision / recall / F1,
  per-class accuracy, confusion matrix, calibration (ECE, Brier, NLL; raw and
  temperature-scaled), accuracy using only the first t time steps, decision
  step, inference time, latency, throughput, peak memory and energy, and the
  same at fewer time steps (T = 5, 10, 15, 25).
- **Explainability:** faithfulness (test accuracy after removing the top,
  bottom or random credited hidden neurons), credit stability and sparsity.

### 6. LaTeX tables

```bash
python scripts/latex_tables.py runs/<compare-dir>/comparison.json -o docs/experiments/<name>.tex \
    --gpu "one NVIDIA RTX 5090"      # hardware named in the cost-table caption
```

Requires `\usepackage{booktabs,multirow,amsmath}`. The tables for the
2026-10-21 comparison are in
`docs/experiments/results_mnist_3seeds_2026-10-21.tex`.

### Single runs

```bash
python scripts/run_snn.py --model ng_snn --dataset mnist --epochs 50       # train + test one model
python scripts/run_snn.py --model bp_snn --dataset mnist --epochs 50
python scripts/run_snn.py --mode eval --ckpt runs/<run>/checkpoints/best.pt  # re-evaluate a checkpoint
python scripts/run_snn.py --model ng_snn --epochs 1 --max-train-batches 50   # quick smoke run
```

Any configuration field can be overridden with `--set key=value` (see
`src/neuron_games/config.py`). `--device` defaults to `cuda` and never falls
back silently.

## Results so far

MNIST, 50 epochs, 3 seeds, obtained on a single NVIDIA RTX 5090 (full tables:
`neuron-games/docs/experiments/results_mnist_3seeds_2026-10-21.tex`):

| | backprop SNN | no-backprop SNN (Neuron Games) |
|---|---|---|
| test accuracy | 97.62 ± 0.08% | 94.19 ± 0.08% |
| training time | 387 s | 388 s |
| training GPU energy | 35.8 kJ | 36.3 kJ |
| peak training memory | 41 MiB | 47 MiB |
| hidden firing rate | 38.9% | 12.0% |
| autograd calls in training | 21,450 per run | 0 |

## Design decisions

Decisions taken while building the no-backprop pipeline, with the reasoning
behind each. Change them only after discussing with the group; other parts of
the code are written against them.

### Learning rule and credit assignment

1. **Every term of `Phi` is trained through the eligibility trace; nothing
   runs a backward pass through the network.**
   For our LIF cell, the factorized eligibility trace `e_ij[t] = h_i[t] * eps_j[t]`
   is exactly the local derivative `ds_i[t]/dW_ij`, and `eps_j[t]` alone is
   `du_i[t]/dW_ij` (verified in `tests/test_learning_rule.py` to float64
   precision). So any term that depends on a neuron's spikes or membrane only
   has to supply a per-neuron signal, and the same rule applies it. This keeps
   the method strictly three-factor, as the proposal requires (section 7.3).
   Every term's signal is computed in closed form; autograd is never called
   during no-backprop training. This is enforced, not assumed: the network's
   weights are frozen out of autograd, each training step runs inside a guard
   that raises on any `backward` / `grad` call, and every run records the
   number of autograd calls in `results.json` (the run fails if it is not 0).
   The tests check each closed form against autograd.

2. **Hidden layers learn from fixed random feedback.**
   The output layer's error is broadcast to hidden neurons through a random
   matrix `B` drawn once and never trained (e-prop with random feedback, the
   proposal's reference [4]). Using the transpose of the output weights instead
   would require weight transport, the main non-local ingredient of
   backpropagation. `B` uses its own random seed, so changing it never changes
   the weight initialization.

3. **The output error is computed in closed form at the output neurons.**
   `dJ/ds_out[t]` for the cross-entropy losses is written out explicitly
   (`potential/task.py`) and checked against autograd in the tests. It is a
   quantity available locally at the output layer, which is what the third
   factor is supposed to be.

4. **No bias terms in the three-factor rows.**
   Every trainable parameter is then a synapse `w_ij` with an eligibility
   trace, which keeps the rule uniform. The config rejects `bias=True` for the
   three-factor rule.

### Optimization

5. **SGD, not Adam, for every three-factor condition.**
   Adam rescales each weight's update by that weight's own gradient history.
   A per-neuron gain from Eligibility-Shapley credit is largely cancelled by
   that rescaling, so the credit modulation would have almost no effect. Plain
   SGD is also the literal form of the proposal's update `dW = eta * int m e dt`
   (eq. 3). The BPTT reference keeps Adam (lr 1e-3).

6. **Default learning rate 2.0, halved on plateau.**
   2.0 was chosen from a 300-batch MNIST sweep over {0.1, 0.5, 2, 5, 10, 20, 50}
   (10 and above diverge) and confirmed with the energy terms on
   (`docs/experiments/followup_c_lr_2026-10-17_en.md`). The learning signals are
   true batch-mean derivatives of small spike-based losses, so a learning rate
   this large is expected under plain SGD. For every model, the learning rate
   halves after 5 epochs without a new best validation accuracy (floor: 1/64
   of the initial value), using the validation split only.

### The potential and its terms

7. **One sign convention for all terms.**
   Each term returns plain derivatives of its own value; the composer
   (`potential/potential.py`) applies the coefficient and sign from
   `C = -Phi = J - beta*V + mu*E + lambda*R`. Diversity `V` is a reward, so it
   enters with `-beta`; energy and the regularizer are costs. Keeping signs in
   one place prevents each contributor from getting them wrong independently.

8. **Every term is summed over neurons and time steps and averaged over the batch.**
   The task loss `J` is a batch mean, so this keeps `beta`, `mu` and `lambda`
   independent of batch size and on a comparable scale to `J`.

9. **Diversity is normalized before the DPP log-determinant.**
   The kernel matrix is divided by the mean of its diagonal before
   `log det(K + eps*I)`. Without this, the log-determinant can be increased
   simply by making every neuron fire more, which rewards activity rather than
   diversity. The jitter `eps` is relative for the same reason.

10. **Plasticity-signature diversity uses an exact shortcut instead of
    materializing the traces.**
    For linear kernels, the similarity of two neurons' eligibility traces is
    `sum_{t,b} h_i h_k ||eps[t,b]||^2`. Feature rows `h_i[t,b] * ||eps[t,b]||`
    give the same Gram matrix at the cost of a spike-based feature, instead of
    a `[neurons x batch x time x inputs]` tensor.

11. **Diversity and credit apply to hidden layers only, by default.**
    Output neurons are class-specific by design; pushing them apart or
    reweighting their task signal would interfere with the read-out.

12. **Barriers start as quadratic penalty plus projection; the log-barrier
    uses the Kervadec extension.**
    A pure log-barrier on every membrane value at every time step is stiff and
    produces infinities in float32 as soon as a value leaves the box. The
    quadratic form is numerically safe for the first runs; the log-barrier
    extension (Kervadec et al. 2019) is finite everywhere and keeps the
    proposal's constrained-game interpretation (eq. 17).

### Eligibility-Shapley credit

13. **Credit uses each neuron's previous update, not the current one.**
    `phi_i = <G_i, dW_i(previous window)>`, with `G` the current sensitivity of
    `Phi`. Using the current update would make the gain depend on the very
    update it scales, and reduces `phi_i` to the squared size of the neuron's
    gradient. With the previous update, `phi_i` is each neuron's first-order
    share of the actual improvement in `Phi`, and the credits sum to the
    first-order reward estimate (the Shapley efficiency property).

14. **Gains are rescaled to average 1 and computed on standardized credit.**
    The proposal's eq. 23 normalizes gains to sum to 1; we multiply by the
    layer size so the mean gain is 1. Uniform credit then reproduces plain
    random feedback exactly, and the difference from eq. 23 is a constant
    absorbed into the global signal. Credit is divided by its standard deviation
    before the softplus because raw values are tiny (a gradient times an
    update), which would otherwise make every gain about 1.

15. **The first window gets uniform credit.**
    There is no previous action yet, so no neuron can be credited.

### Experiment setup

16. **One adaptation window is one mini-batch presented for T = 25 steps.**
    Weights are updated once per window, after the rollout, matching the
    proposal's windowed algorithm (section 11). The traces are computed step by
    step, so online updates within a window remain possible later.

17. **The no-backprop rule and the BPTT reference train the identical network.**
    Same LIF cell, same 784-128-10 architecture, same encoding, loss, data split
    and seeds, so a difference in results is attributable to the learning rule.

18. **Shared measurement code is imported, not copied.**
    `neuron-games` imports the data loaders, seeding, NVML energy meter, plots
    and kernel library from `snn-surrogate-baseline`, so both pipelines are
    measured by the same machinery.

19. **Ablation conditions are presets, and every term can be toggled
    individually.**
    `--condition` selects a row of the ablation grid (proposal section 12.3);
    any term can also be switched on alone on top of `three_factor`. This lets
    each part be developed and tested before the others are finished.

### Measurement

20. **Every metric runs for every model.**
    Training diagnostics (Phi, credit, gradient and update norms), test metrics
    and explainability are computed identically for the backprop and
    no-backprop SNN, and for any model added later, unless a metric is
    genuinely incompatible with a model (the report then says so). For the
    backprop SNN, Phi and credit are measured, never trained on.

21. **Training, validation and test are timed separately, with GPU
    synchronization at every boundary.** Asynchronous GPU work would otherwise
    be charged to the wrong phase. Time-to-accuracy counts training time only.

22. **Peak memory is measured per phase, with cuDNN autotuning off.** The
    baseline's seeding turns autotuning on for its conv net; here it only added
    a one-time 0.5-1 GB scratch allocation in the diversity term's 1-D
    smoothing, which distorted the memory numbers. The training model is freed
    before test inference so inference memory is measured on its own.

23. **Calibration is reported after temperature scaling.** With the `ce_rate`
    loss, the model's raw class probabilities are squashed toward 1/10, so raw
    calibration error mostly measures that definition. The report leads with
    temperature-scaled numbers (one temperature fitted on the validation split
    per model) and keeps the raw ones.

## Log

- 2026-09-07 - Repo initialized; proposal read and summarized.
- 2026-09-23 - Reorganized into workspace-of-projects layout. Added
  `snn-surrogate-baseline/` (Tutorial 6 BP baseline on RTX 5080).
- 2026-10-08 - `neuron-games`: vendored LIF cell, 784-128-10 network, BPTT reference row.
- 2026-10-14 - `main` set up from the `joohyun` branch. Three-factor core of
  Neuron Games implemented (trains on MNIST without backprop, 91.8% test after
  3 epochs); remaining components assigned in `Architecture_Work_14-10-2026.md`.
- 2026-10-15 to 2026-10-19 - Diversity (Deyvik), homeostasis and ablation driver
  (Joohyun), Eligibility-Shapley credit and explainability (William), and the
  Shapley-Taylor stretch (Deyvik) implemented and merged.
- 2026-10-21 - No-backprop training made strictly autograd-free and audited;
  full metric suite for every model; single run script and comparison
  pipeline; first 3-seed MNIST comparison (results above).
