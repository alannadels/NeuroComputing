# Batch __ID__ plan: __TITLE__

Written before the batch runs; not edited afterwards (record surprises in the
analysis in `EXPERIMENT_LOG.md` instead).

- author: <name>
- date: <YYYY-MM-DD>
- code: commit `<short hash>` (the batch tags it `exp/batch-__ID__`)
- base: <configuration and where it comes from, e.g. "current best, docs/EXPERIMENT_SUMMARY.md">
- reference: <backprop control ID(s)>

## Why this batch

<What the previous results (yours or docs/EXPERIMENT_SUMMARY.md) leave open, and
which front(s) this batch targets: accuracy, energy, efficiency, explainability.
Check the "What did not work" table first so nothing is repeated.>

## Experiments

| ID | change vs base | reference | hypothesis (what result would confirm or refute it) |
|---|---|---|---|
| C01-C03 | backprop control, seeds 1-3 | -- | -- |
| B01-B03 | base, seeds 1-3 | C01 | reproduces the summary result |
| E01 | <one change> | C01 | <expected effect and why> |

## Decision rule

<How the outcome decides the next step, decided before seeing results. Select
by validation accuracy only; single-seed differences below ~0.3 pt are noise,
so confirm anything promising on seeds 2 and 3 before adopting it.>
