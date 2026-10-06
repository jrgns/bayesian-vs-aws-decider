# Plan — Determinism and stability test (NB vs Decider)

Status: **APPROVED 2026-10-06. BLOCKED on the main benchmark.** Decisions: D6 runs in both forms (D6a, D6b), separate processes, reweighted flip rate is the headline.

## Dependency (why this can't run yet)
`nb-vs-decider/` currently holds only the step-1 scaffolding: Docker, pins and the Decider smoke test. There is no
`REPORT.md` yet. This task needs these outputs of the main run ([PLAN.md](PLAN.md)):

| Needed | Produced by main plan step |
| --- | --- |
| Frozen test split + item IDs | 3 |
| Frozen Decider label wording | 4 |
| Decider per-item top-2 margins (main test run) | 4 |
| Throughput batch size | 4 |
| NB best config at the crossover size | 5–6 |
| `REPORT.md` to link back into | 8 |

The stability work starts once the main `REPORT.md` exists. Nothing is changed: same image, lockfile, dataset, split,
Decider release and label wording. In this project, "reuse the venv" means reuse the same Docker image and `uv.lock`.

## Stability subset (1,000 items, seed recorded, IDs saved)
- **Margin** = top-1 minus top-2 option probability from the main Decider test run (batch size 1).
- **Near-boundary half (500):** drawn from the lowest-margin decile of the 10k test set (1,000 items). Stratified by
  true label proportionally to that decile's class mix. Low-margin items won't be class-balanced, and forcing
  balance would distort the sample.
- **Rest (500):** stratified and class-balanced, drawn from the other 9,000 test items.
- Saved to `stability/subset_ids.csv` with a `stratum` column.
- Every metric is reported for all 1,000, for near-boundary and for rest. **Proposed addition:** a reweighted
  test-set estimate (near-boundary weighted 10%, rest weighted 90%). The 1,000-item subset over-samples hard items,
  so its raw flip rate overstates the real one. The reweighted figure is the number that goes next to accuracy in
  `REPORT.md`.

## Conditions
Unless noted, **each run is a separate process** (`docker compose run`, fresh model load). That matches realistic
serving and also catches load-time nondeterminism. Item order is canonical unless the condition changes it.

| ID | What | Runs | Notes |
| --- | --- | --- | --- |
| N1 | NB, one trained model, predict 10× in one process | 10 | Crossover-size best config, training subset of seed 0 |
| N2 | NB, retrain (same subset, same fixed config, no re-tuning) + predict, separate processes | 10 | `PYTHONHASHSEED` left at its default random value to make it realistic |
| D1 | Decider, batch size 1, default PyTorch | 10 | |
| D2 | Decider, throughput batch size, default | 10 | Control for D5 and D6 |
| D3 | D2 + `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.benchmark=False`, fixed seeds | 10 | Env and flags set **before** CUDA initialises. If an op raises because it has no deterministic implementation, that is recorded as a finding. Then it is re-run with `warn_only=True`, labelled D3w |
| D4 | Batch size 1 vs throughput batch size | 0 new | Compares D1 run *i* with D2 run *i*, plus the flip rate over the pooled D1 ∪ D2 runs |
| D5 | Option order shuffled per run (seed = run index) at the throughput batch size | 5 | Mapped back to canonical labels. Read against D2 as baseline |
| D6a | Item order permuted **within each batch** (same batch composition), throughput batch size | 5 | |
| D6b | Whole subset shuffled before batching (different batch neighbours and padding), throughput batch size | 5 | Realistic serving variation |

If NB never reaches Decider (no crossover), N1 and N2 use the best NB config at the full pool, and this is stated.

## Metrics (per condition × stratum)
- **Flip rate:** share of items whose predicted label is not identical across all runs.
- **Pairwise disagreement:** mean over all run pairs.
- **Bitwise repeatability:** share of items whose raw scores are byte-identical across runs (`float32` bytes),
  plus the max absolute score difference.
  - Raw score = the pointer-head logits if the API exposes them, otherwise the per-option probabilities.
    The smoke test (step 1) is recording which one is available.
- Accuracy min–max across runs.
- Wall-clock time, items/s, and p50/p95 latency. These make the speed cost of D3 vs D2 visible. The GPU-idle
  preflight runs first, but no energy is measured, since it's not requested.

## NB flips
No flips are expected. On this path the vectorizer vocabulary is sorted, and NB prediction is a sparse × dense
product. If any flip or bit difference shows up, the investigation goes in this order before anything is reported:
1. thread count / BLAS: `OMP_NUM_THREADS`, `threadpoolctl` info
2. hash-seed or dict-order dependence
3. float summation order

## Outputs (`nb-vs-decider/stability/`)
- `run_stability.sh`: one command. Runs preflight → N1, N2, D1–D6 → metrics → `STABILITY.md`.
- `runs/<condition>/run_<k>.parquet`: item_id, label, prediction, per-option scores, run metadata.
- `results.csv`: one row per condition × stratum.
- `STABILITY.md`:
  - summary table
  - which conditions are bit-for-bit repeatable
  - causes of flips (batch size / kernels / option order / batch position)
  - 5 flipped items with their margins
  - recommended repeatable production config and its throughput cost
  - versions: torch, CUDA, driver, scikit-learn, numpy, Decider commit and model revision
- A "Stability" section appended to `REPORT.md`: flip rate (reweighted) next to accuracy and kWh, with a link.

## Ground rules applied
- Nothing is tuned to reduce flips. Default conditions are measured as configured, and the deterministic config is
  reported separately.
- GPU-idle preflight before any Decider run. If llama-swap/vLLM holds the GPU, **stop and ask**. Never kill anything.

## Estimated cost
About 1,000 items × (10 + 10 + 10 + 5 + 5 + 5) runs. At roughly 100 ms per item at batch size 1 and much faster when
batched, that is roughly **45–75 min of GPU time** on one 3090. NB takes seconds.

## Decisions (2026-10-06)
1. D6: both readings are run, as D6a (within-batch) and D6b (whole-subset shuffle).
2. D1–D3: each run is a separate process.
3. The reweighted flip rate (near-boundary 10% / rest 90%) is the headline figure in `REPORT.md`.
