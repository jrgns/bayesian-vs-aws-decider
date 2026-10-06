# Stability: are NB and Decider repeatable?

This is part of the [main report](../REPORT.md); the plan is in [`docs/PLAN_STABILITY.md`](../../docs/PLAN_STABILITY.md).
Nothing was tuned to reduce flips. Everything is the same as the main run: image, lockfile, dataset, test split, Decider
release and label wording v3. Every Decider run went through the official `collect_logits` path, as its own
`docker compose run` process, on one RTX 3090 (orion). Run everything with `ALLOW_TEST=1 stability/run_stability.sh`.

## Subset

1,000 test items (seed 20261007, `subset_ids.csv`):
- **near_boundary (500):** sampled from the 1,000 test items with the lowest `margin_top2` in the main bs=1 run (margins
  0.0001 to 0.110), stratified proportionally by true label.
- **rest (500):** class-balanced from the other 8,996 items (72 for each of the first three labels, 71 for the rest).
  Median margin 0.654.

The subset over-samples hard items, so Decider's accuracy on it (0.59) is below its full-test 0.7207. The headline figure
is **reweighted**: near_boundary counts for 10% and rest for 90%, which estimates the full test set. Flip rates are additive,
so the reweighted figure is the weighted mean of the two strata.

**Definitions:**
- **Flip:** the item's predicted label is not the same in every run.
- **Bitwise identical:** the item's raw float32 logits are byte-identical across all runs. For NB, the comparison uses
  `predict_proba` (float64).
- **Run order:** the canonical item order is a seeded shuffle of the subset. Without it, consecutive pairs (bs=2) would always
  pair near-boundary items with near-boundary items.

## Summary table

Stratum = all, except the last column (reweighted flip rate).

| Condition | Runs | Flip all | Flip near | Flip rest | Pairwise disagr. | Bitwise identical | Max abs diff | Acc min–max | Items/s | Wall s | Flip, reweighted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N1 NB, one model, predict 10× | 10 | 0 | 0 | 0 | 0 | 1.000 | 0 | 0.680 | 7212 | 0.139 | **0** |
| N2 NB, retrain in each process | 10 | 0 | 0 | 0 | 0 | 1.000 | 0 | 0.680 | 7180 | 0.140 | **0** |
| N1 + N2 pooled | 20 | 0 | 0 | 0 | 0 | 1.000 | 0 | 0.680 | 7196 | 0.139 | **0** |
| D1 bs=1, default | 10 | 0 | 0 | 0 | 0 | 1.000 | 0 | 0.588 | 9.87 | 101.4 | **0** |
| D2 bs=2, default | 10 | 0 | 0 | 0 | 0 | 1.000 | 0 | 0.587 | 10.69 | 93.5 | **0** |
| D3 bs=2, deterministic mode | 10 | 0 | 0 | 0 | 0 | 1.000 | 0 | 0.587 | 9.91 | 100.9 | **0** |
| D4 D1 + D2 pooled | 20 | 0.015 | 0.030 | 0 | 0.0079 | 0 | 0.058 | 0.587–0.588 | 10.28 | 97.4 | **0.0030** |
| D4 paired (D1 run *i* vs D2 run *i*) | 10 | 0.015 | 0.030 | 0 | 0.015 | 0 | 0.058 | 0.587–0.588 | | | **0.0030** |
| D5 option order shuffled | 5 | 0.413 | 0.720 | 0.106 | 0.2005 | 0 | 1.96 | 0.525–0.600 | 10.54 | 94.9 | **0.167** |
| D5 + D2 pooled | 15 | 0.436 | 0.764 | 0.108 | 0.129 | 0 | 1.96 | 0.525–0.600 | 10.64 | 94.0 | **0.174** |
| D6a pair swapped within each batch | 5 | 0 | 0 | 0 | 0 | 1.000 | 0 | 0.587 | 10.57 | 94.7 | **0** |
| D6a + D2 pooled | 15 | 0 | 0 | 0 | 0 | 1.000 | 0 | 0.587 | 10.65 | 93.9 | **0** |
| D6b whole subset shuffled | 5 | 0.012 | 0.024 | 0 | 0.0060 | 0.189 | 0.056 | 0.582–0.589 | 10.59 | 94.4 | **0.0024** |
| D6b + D2 pooled | 15 | 0.014 | 0.028 | 0 | 0.0040 | 0.163 | 0.056 | 0.582–0.589 | 10.66 | 93.8 | **0.0028** |

Batch latency p50 / p95 in ms per batch: D1 95.6 / 121.0 (1 item); D2 179.7 / 219.6 and D3 194.4 / 238.1 (2 items).
The full table per stratum, including reweighted accuracy bounds, is in [`results.csv`](results.csv).

**Cross-check against the main run.** The reweighted D4 flip rate, 0.30%, matches the main test run, where bs=1 and bs=2
disagreed on 31 of 9,996 items (0.31%).

## Which conditions are bit-for-bit repeatable

- **Yes:** N1, N2, D1, D2, D3 and D6a. That holds for every item, in every run and every separate process.
- **No:** D4 (bs=1 vs bs=2), D5 and D6b.

With the model, hardware, software stack and run recipe all fixed, logits are identical across fresh processes, even with
default PyTorch settings. Deterministic mode (D3) isn't needed for this. D3 raised no error, because no op lacked a
deterministic implementation, so D3w was not run. D3 output is byte-identical to D2.

## Causes of flips

Every flip comes from a change in the input Decider sees. None comes from run-to-run noise. The evidence is in
[`causes.json`](causes.json).

- **Batch size (D4: D1 vs D2).**
  - 15 items flip, all near the boundary (margin median 0.003, max 0.014).
  - None of the 1,000 items is byte-identical between bs=1 and bs=2. That includes the 498 items that are the longer of their
    pair and so get no padding (0 of 498).
  - So changing the batch size changes the GPU kernel and tiling choice, and with it the float rounding. Logits move by up to
    0.058.
- **Batch neighbours and padded length (D6b).** A batch is padded to its longest text.
  - When an item's batch has the same padded length as in D2, 99.0% of item-runs are byte-identical (1,685 item-runs).
  - When the padded length differs, only 15.4% are (3,315 item-runs).
  - So sequence length drives the kernel numerics. 12 items flip, all near the boundary, with margins up to 0.016 (median 0.005).
- **Position within a batch (D6a).** No effect. Swapping each pair gives output byte-identical to D2. At bs=2, which item comes
  first makes no difference.
- **Option order (D5).** This is the large effect.
  - 41% of items flip: 72% of near-boundary items, and 10.6% of the easy ones too. Accuracy ranges from 0.525 to 0.600.
  - This is a real input change, not rounding. The order of the options is part of the prompt, and Decider has a position
    preference. When Medical Science sits in the first option slot (runs 1–3), the model predicts Medical 283–316 times out of
    1,000, against 192 in canonical order.
  - Max logit difference 1.96. Flipped items have margins up to 0.57.
  - *Re-check (orchestrator):* in each shuffled run, label names and descriptions move together as pairs. Columns are mapped
    back with `out[:, perm] = arr` (unit-tested). Each run still agrees with canonical D2 on 74–80% of items and scores
    0.525–0.600. A mapping bug would have pushed accuracy towards chance (about 14%), so the effect is real.
- **Kernel nondeterminism and process start (D1, D2, D3).** None observed. Fresh processes, cuDNN autotune and cuBLAS defaults
  all gave identical bytes.
- **NB.** No flips and no byte differences.
  - The vocabulary is sorted, BLAS runs on one thread, and the sparse × dense product has a fixed summation order.
  - N2 ran with a random `PYTHONHASHSEED` in each of its 10 processes and was still byte-identical. That shows 10 random hash
    seeds agree; it isn't a formal proof of independence.
  - So the planned investigation (BLAS, hash seed, summation order) had nothing to explain.

Apart from D5, every flipped item is a near-tie, where a tiny numeric change is enough to swap the top two labels.
D5 is different because it changes what the model is shown.

## 5 flipped items

Margin = top-1 minus top-2 probability in the main bs=1 run. Predictions are listed per run; for D4 that is 10 D1 runs
followed by 10 D2 runs. Texts are the first 120 characters of each abstract.

| # | Condition | Item | True | Margin | Predictions | Text |
|---|---|---|---|---|---|---|
| 1 | D4 | `7e9ce4b923f95800` | 3 | 0.0011 | D1: 3 ×10; D2: 4 ×10 | There are significant levels of concern about the relevance and the difficulty of learning some issues on Strength of Ma |
| 2 | D6b | `d0f3df76df8921bf` | 1 | 0.0010 | 1, 3, 3, 3, 3 | The big temperature rising is the principle problem that constrains the application of the permanent magnet synchronous |
| 3 | D5 | `ebfa7c31868255d3` | 4 | 0.0001 | 0, 1, 3, 1, 1 | Ionic Polymer-Metal Composite (IPMC) is a new smart material which has sensing capability. In this paper, an experimenta |
| 4 | D4 | `b8974eb3b39ac98f` | 3 | 0.0011 | D1: 4 ×10; D2: 3 ×10 | Structural systems are subject to uncertainties due to variability in many hard-to-control noise factors, which include |
| 5 | D6b | `a5baf704929a55e2` | 5 | 0.0013 | 5, 5, 6, 6, 5 | Alternative and complementary procedures were adopted for preparing 2-arylazetidine derivatives in moderate to good yiel |

Labels: 0 Computer Science, 1 Electrical Eng., 2 Psychology, 3 Mechanical Eng., 4 Civil Eng., 5 Medical Science, 6 Biochemistry.

## Recommended repeatable config and its cost

- **Use the default Decider settings at bs=2, with a fixed option order and fixed batch composition.** This is bit-for-bit
  repeatable across processes at 10.69 items/s, so repeatability costs nothing here.
- **Adding the D3 flags is optional insurance.** The flags are `torch.use_deterministic_algorithms(True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.benchmark=False` and `cudnn.deterministic=True`, plus seeds, all set before CUDA
  initialises.
  - They cost +7.9% wall time (93.5 s → 100.9 s), i.e. −7.3% throughput (10.69 → 9.91 items/s).
  - They changed no byte on this stack.
  - They don't fix D4 or D6b.
- **If you can't control which items share a batch, use bs=1.** It runs at 9.87 items/s (−7.7% vs bs=2) and was repeatable
  in 10 runs. With no batch neighbours it should be unaffected by them; that wasn't tested separately.
- **bs=1 and bs=2 give different results** (D4), so choose one batch size and keep it.
- **Never vary the option order.** It changes predictions on 41% of this subset (16.7% reweighted).

## Versions

- **Image id:** `sha256:eed86c43defcfcbd11abfe43bdf038d97247ab665d5f7330afe2b3d2d5d57558`. The image was rebuilt to add these
  scripts, with the same Dockerfile and lock as the main run's `sha256:b783694e1ecc954c607dd7d8c9c90850fc870a37bf5e739ea665a855c3b2eab4`.
- **GPU stack:** torch 2.14.1+cu130, CUDA 13.0, cuDNN 9.24, driver 595.91.07, RTX 3090.
- **Python stack:** scikit-learn 1.9.1, numpy 2.5.3, threadpoolctl 3.7.0, scipy 1.18.1, Python 3.12.15.
- **Decider:** strands-decider 0.1.0. Model `StrandsAgents/strands-decider-2B-hobson-v21`, revision
  `2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5`. Base `Qwen/Qwen3.5-2B-Base`, revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`.
  The installed package carries no git commit; the training-data inventory commit is `3e94e9d84c620ed5a95f1a3310c3decb971e261c`.
- **Other:** causal-conv1d 1.7.0; CPU AMD Ryzen 9 3900.
- **Configuration:** labels v3; throughput batch size 2. NB is ComplementNB + CountVectorizer with alpha 0.3 and min_df 2
  (n=100 per class, seed 0).
- **Full list:** [`versions.json`](versions.json).

Back to the [main report](../REPORT.md).
