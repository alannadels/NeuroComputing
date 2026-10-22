# Experiment tracking

How experiments are run, recorded and restored. Experiments are run in
batches of 10 (Iterative Experiment Refinement): each batch is designed from
the results of the previous one, and every batch, configuration and result is
kept so that any version of the model can be inspected or rerun later.

## Layout

The `experiments/` folder lives next to the code on the machine that runs the
experiments and is git-ignored (large outputs, checkpoints). Its scripts and
reports are the record of the project's experimental history.

```
neuron-games/experiments/
├── VERSIONS.md                 version registry: every model version that ran
│                               (commit + tag + settings + key results + restore command)
├── EXPERIMENT_LOG.md           master log: per batch, the plan, the results, the analysis,
│                               and the decisions that led to the next batch
├── batch_NN/
│   ├── PLAN.md                 hypotheses and exact settings, written before running
│   ├── run_batch_NN.sh         the script that ran the batch
│   ├── batch_meta.json         commit, tag, defaults, start time
│   ├── env/                    source snapshot (tar.gz of the commit), packages, GPU, test results
│   ├── batch.log               full console log
│   ├── <ID>_<slug>/            one experiment: experiment.json (exact command, overrides,
│   │                           commit, status), config.json (fully resolved settings),
│   │                           results.json, explain.json, checkpoints/, plots/, train.log
│   ├── BATCH_REPORT.md         generated: what ran, results, every front vs the reference
│   └── batch_results.json      the same, machine-readable
└── _restore/                   worktrees and reruns created by exp_reproduce.sh
```

## Rules

1. **No experiment runs on uncommitted code.** `batch_init` refuses to start if
   tracked files differ from the commit. Every result therefore maps to an exact
   commit, which is tagged `exp/batch-NN` and also archived inside the batch
   (`env/source_<commit>.tar.gz`).
2. **Every experiment records its exact command and settings** (`experiment.json`)
   and its fully resolved configuration (`config.json`). Changing a setting is a
   new experiment with a new ID, never an edit of an old one.
3. **Each batch includes a backprop control** (`C01`, the backprop SNN with
   default settings on the batch's commit), so every experiment is compared
   with a reference run on identical code. Experiments that change something
   both models share (for example the read-out) also get a matching backprop
   run and use it as their reference (`--ref`).
4. **Results are appended, never edited.** `VERSIONS.md` and `EXPERIMENT_LOG.md`
   grow by one section per batch.

## Running a batch

```bash
source <your-venv>/bin/activate
cd neuron-games
tmux new -s batchNN
experiments/batch_NN/run_batch_NN.sh
```

A batch resumes where it stopped if it is restarted (finished experiments,
those with a `results.json`, are skipped).

## Going back to any version

```bash
# rerun an experiment exactly (same commit, same command, same seed), into a new run directory
scripts/exp_reproduce.sh experiments/batch_03/E07_<slug>

# only check out that exact code, to inspect it or branch from it
scripts/exp_reproduce.sh experiments/batch_03/E07_<slug> --checkout-only

# the code of a whole batch
git checkout exp/batch-03
```

`VERSIONS.md` lists every version with its commit, settings and restore
command. Version v0 is the baseline before experimentation began.

## The four fronts

Each experiment is scored against its reference on:

| front | metrics (better) |
|---|---|
| accuracy | test accuracy (higher) |
| energy | training GPU energy (lower); synaptic operations per test image caused by the network's own spikes (lower; input-spike operations are identical for every model) |
| efficiency | training time per epoch, peak training memory, inference latency, epochs to 95% validation accuracy (lower) |
| explainability | faithfulness margin: accuracy after removing 64 random hidden neurons minus after removing the 64 top-credited ones, best of the two credit scores (higher; positive = faithful) |
