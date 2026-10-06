# Naive Bayes vs AWS Strands Decider 2B

How many labelled examples does a classic scikit-learn Naive Bayes classifier need to match
the **zero-shot** accuracy of [AWS Strands Decider 2B](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)
on a text-classification task (WOS-46985, 7 classes), and what does each approach cost in energy?

## Headline results
- **Crossover:** NB matches Decider (0.721 accuracy) at about 70 examples per class (490 labels; refined grid, a statistical tie).
  On the pre-registered grid the crossover is 100 per class (700 labels): NB scores 0.748 and is significantly better (McNemar, 5/5 seeds).
- **Inference energy:** Decider 8.2 Wh per 1,000 predictions (GPU + CPU package, gross); NB 1.5 mWh (CPU only), about 5,600x less.
- **Break-even:** NB's training energy (with tuning) is repaid after about 7 predictions at 70/class, or 10 at 100/class.

Full write-up, tables, threats to validity and versions: [`nb-vs-decider/REPORT.md`](nb-vs-decider/REPORT.md).
Plots: `nb-vs-decider/plots/learning_curve.png`, `nb-vs-decider/plots/energy.png`. Plan and decision log: [`docs/PLAN.md`](docs/PLAN.md).

## Running
Runs in Docker on a Linux host with an NVIDIA GPU and the NVIDIA container toolkit (reference host: 2x RTX 3090, GPU 0 is used).
CPU energy counters (RAPL) are bind-mounted read-only into the container; no host permission changes are needed.
```bash
cd nb-vs-decider
ALLOW_TEST=1 ./run_all.sh            # dataset -> contamination -> Decider -> NB curve -> crossover -> energy -> plots
DRY_RUN=1 ALLOW_TEST=1 ./run_all.sh  # only print each stage as skip/RUN
```
Each stage skips when its output file exists (delete it to rerun). `preflight_host.sh` aborts if the GPU is busy and never stops
services. `ALLOW_TEST=1` unlocks every stage that touches the test split and is never set by a script. Lint and tests:
`docker compose run --rm dev`. The analysis stages (`crossover`, `energy_summary`, `plots`) are CPU only and read `results/`.

## Layout
`nb-vs-decider/scripts/` holds one module per stage, `config/labels/` the versioned Decider label wording (`LOG.md` has the
selection rule), `results/` the raw outputs, and `plots/` the figures.

## Contributors
Agent instructions are in [`AGENTS.md`](AGENTS.md).
