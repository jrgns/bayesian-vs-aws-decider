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
Runs in Docker on a Linux host with an NVIDIA GPU and the NVIDIA container toolkit (reference host: 2× RTX 3090).
```bash
cd nb-vs-decider
./run_all.sh        # end-to-end: data → contamination check → Decider → NB curve → energy → report
```
`run_all.sh` runs a host GPU-idle preflight, then the pipeline in the `nb-vs-decider` image. CPU energy counters are bind-mounted read-only into the container. No host permission changes are needed.

## Contributors
Agent instructions are in [`AGENTS.md`](AGENTS.md).
