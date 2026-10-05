# Naive Bayes vs AWS Strands Decider 2B

How many labelled examples does a classic scikit-learn Naive Bayes classifier need to match
the **zero-shot** accuracy of [AWS Strands Decider 2B](https://huggingface.co/StrandsAgents)
on a text-classification task, and what does each approach cost in energy?

## Status
Planning. See [`docs/PLAN.md`](docs/PLAN.md). No benchmark has been run yet.

## What it will produce (in `nb-vs-decider/`)
- **Crossover point:** the training-set size at which NB is statistically level with Decider (McNemar test).
- **Energy:** kWh per 1k predictions for both methods, plus NB training energy. Measured with
  NVML (GPU) and RAPL (CPU) hardware counters, minus an idle baseline.
- `results/*.csv`, `plots/learning_curve.png`, `plots/energy.png`, `REPORT.md`.

## Running (once implemented)
Designed for a Linux host with an NVIDIA GPU (reference host: 2× RTX 3090).
```bash
cd nb-vs-decider
./run_all.sh        # end-to-end: data → contamination check → Decider → NB curve → energy → report
```
Reading CPU energy counters (RAPL) usually needs extra permissions on Linux ≥ 5.10.

## Contributors
Agent instructions are in [`AGENTS.md`](AGENTS.md).
