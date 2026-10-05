# Plan — Naive Bayes vs AWS Strands Decider 2B

Status: **APPROVED 2026-10-05** (dataset WOS-46985 L1; everything runs in Docker). Nothing long-running has been executed.
Written 2026-10-05.

## 0. Findings so far (verified 2026-10-05)

**Decider exists and moves fast.** Released 2026-10-01 by AWS Strands Labs.
- Model: [`StrandsAgents/strands-decider-2B-hobson-v21`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)
  (rev `2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5`, updated 2026-10-05). The previous release, v19, is rev `bb282d78…`.
  It is a LoRA adapter on `Qwen/Qwen3.5-2B-Base` (rev `b1485b2f…`) plus a pointer head that scores the supplied options. It does not generate text.
- Code: [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider), `main` at `3e94e9d8…`. There are no git release tags yet.
- PyPI: `strands-decider==0.1.0`. It needs `torch>=2.7` and `transformers>=5.15,<6`.
- Inference: `StrandsDeciderModel.load(...)`, a `choice` question type that returns per-option probabilities, the CLI (`strands-decider ask`) and an HTTP server (`/v1/systemone`).
- **Pinning risk:** v21 ships `strands_decider_config.json`, while v19 ships `hobson_config.json`. PyPI 0.1.0 may only load v19. Step 1 checks this. We pin whichever pair (package, model revision) the official code loads cleanly.

**Contamination by dataset name: every default candidate fails.** The model card's `datasets:` list and
[`data/sources.md`](https://github.com/strands-labs/strands-decider/blob/main/data/sources.md) contain:

| Candidate | In Decider's data? | Verdict |
| --- | --- | --- |
| `yahoo_answers_topics` | Yes. Recipe `yahoo_topics`, role **train** | Rejected |
| 20 Newsgroups | `SetFit/20_newsgroups` is in the card's dataset list. It is not in the current recipe table, so it was probably used by an earlier recipe | Rejected (conservative) |
| DBpedia-14 | Yes. Recipe `dbpedia`, **train** | Rejected |
| AG News | Yes. Recipe `ag_news`, **train** | Rejected |

Other excluded datasets: arxiv-classification, banking77, clinc150, massive (intent), emotion, TREC, SST-5, Yelp, civil_comments and the rest of the card list.

**Proposed replacement (needs your OK): Web of Science `WOS-46985`, level-1 labels**
([`HDLTex/web_of_science`](https://huggingface.co/datasets/HDLTex/web_of_science), rev `26ae29c3…`).
- 7 classes with plain-English names: Computer Science, Electrical Engineering, Psychology, Mechanical Engineering, Civil Engineering, Medical Science, Biochemistry.
- Texts are scientific abstracts of about 200 words. That is enough for bag-of-words and far inside Decider's 4096-token window.
- 46,985 documents, which should give ≥ 1k per class. The class sizes are imbalanced, so we sample balanced test and dev splits. Step 2 verifies the counts.
- The dataset is not in Decider's list by name. The n-gram check below must still pass.
- Downsides: it is less famous than the defaults, and it is an older script-based HF dataset. If it won't load with a modern `datasets` version, we use a pinned parquet conversion and record its hash.

Fallbacks if WOS fails (to be verified, in order): an older or broader news-topic corpus outside the list, then a 5–15-class collapse of a larger taxonomy. Each one goes through the same checks.

**Hosts.** This workstation is `nebula` (i7-1165G7, no NVIDIA GPU). All measured runs happen on **orion**. The scripts are developed here and executed there.

**orion, verified 2026-10-05:**
- AMD Ryzen 9 3900 (12 cores), kernel 7.0.0-34, NVIDIA driver 595.91.07, 2× RTX 3090. Both GPUs were idle (1 MiB used).
- Docker 29.4.2 with the `nvidia` runtime.
- RAPL is exposed as `intel-rapl:0` (`package-0`). Its `energy_uj` is root-only (`-r--------`).

## Execution environment: Docker (approved)
Everything except a host preflight runs in one image, `nb-vs-decider`. That covers the data, the contamination check, Decider, NB, energy, plots and the report.
- **Image:** `nvidia/cuda` base image with a `uv`-created venv at `/opt/venv`. Every package is pinned from `uv.lock`, and `requirements.txt` and `requirements-dev.txt` are exported from it. Decider is installed at the pinned version.
- The image is built for `linux/amd64` and `linux/arm64` with `docker buildx`. There is no GitLab remote, so it is built locally for now.
- **compose services:**
  - `pipeline`: GPU 0 only (`device_ids: ["0"]`). Bind mounts `results/`, `plots/` and the report. Has a named volume for the HF cache.
  - `dev`: CPU only. Runs ruff and pytest, and works on nebula.
- **RAPL in Docker (tested on orion):** Docker masks `/sys/devices/virtual/powercap` by default as a PLATYPUS mitigation, so reading it in a default container fails. A read-only bind mount (`/sys/devices/virtual/powercap:/host/powercap:ro`) can be read by the container's root. That needs no `privileged` mode, no AppArmor change and no host chmod.
- **NVML in Docker (tested):** `--gpus device=0` exposes only the measured GPU.
- **Host preflight (`preflight_host.sh`):** run by `run_all.sh` on orion before `docker compose run`. It checks that `nvidia-smi` shows no compute processes and that llama-swap/vLLM are not holding a GPU. If they are, it **aborts and asks**. It never stops services. Inside a container, `nvidia-smi` can't see host process names, which is why this check runs on the host.
- **Caveat:** Docker isolates the software, not the energy. RAPL and NVML count for the whole host and the whole GPU. The idle baseline is measured in the same container setup, so the daemon and container overhead cancel out in the marginal figures. The Decider and NB containers are pinned to fixed CPU sets (`cpuset`) so runs are repeatable.

## 1. Pin and smoke-test Decider (on orion, short run)
1. Build the image (pinned `uv.lock` → venv at `/opt/venv`). Exported `requirements*.txt` are committed.
2. Install `strands-decider` at the pinned version. Load the pinned model revision. Run the model-card example and confirm it matches the published output to about 3 decimals.
3. Read `docs/inference.md` in the repo. Use the reference `choice` path. Find the max-options and context limits and the batching API.
4. Record versions: package, model SHA, git SHA, base model SHA, torch, CUDA, driver, CPU.

→ verify: card example reproduces; `versions.json` written.

## 2. Dataset decision and contamination check (report section)
1. Name check: the dataset id against the model-card list and the `sources.md` tables.
2. Text overlap, exact and 13-gram, between our test set and Decider's corpus:
   - the committed synthetic files (`data/synthetic/*.jsonl`, `data/generators/gen_*/`), which are exact files;
   - the 21 short-task and 7 held-out HF datasets, plus ContractNLI, MuSiQue, BoardgameQA, HotpotQA and HelpSteer2, rebuilt via `load_dataset`. The pins are not recorded upstream, so this is approximate.
   - Metric: the share of test items with any shared 13-gram after normalisation (lowercase, whitespace, punctuation), plus the exact-match count. Threshold: < 1% of items. Any hit is listed.
3. Record class counts and token-length distribution with Decider's tokenizer. Confirm 0% truncation.
4. Known limitation: the Qwen3.5 base pre-training corpus is not published and cannot be checked. This goes under threats to validity.

→ verify: `results/contamination.csv`, a decision paragraph, and why each rejected candidate was rejected.

## 3. Splits (fixed, seed recorded)
- **Test:** 10,000 items, stratified and balanced, drawn from the official test split where one exists. Frozen to `data/splits/test_ids.txt` with a hash.
- **Dev:** 2,000 items, balanced, disjoint from test. Used for Decider label wording only.
- **Training pool:** everything else. Each NB subset is sampled from this pool; NB CV runs inside the subset.

## 4. Decider zero-shot
1. Label descriptions v1 are written before any model output is seen and committed to `config/labels/v1.yaml`.
2. Optional rewording iterations scored on **dev only**. Each version is logged in `config/labels/vN.yaml` with its dev accuracy. Wording changes only, no examples. Final version frozen before test.
3. One test run at batch size 1. Then a batch-size sweep (1, 2, 4, …) on a dev subset to find the best throughput, followed by one test run at that size, with predictions checked to be identical within tolerance.
4. Metrics: accuracy, macro-F1, 95% bootstrap CI (10k resamples), Brier score (multiclass), ECE (15 bins). Per-item predictions are saved for McNemar.

## 5. NB learning curve
- Examples per class: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, full pool. 5 seeds each.
- Grid: {Multinomial, Complement}NB × {Count, Tfidf}Vectorizer(ngram_range=(1,2)) × `alpha` ∈ log grid × `min_df` ∈ {1, 2, 3}.
- Model selection by stratified k-fold CV inside the training subset, with k = min(5, n_per_class). For n = 1 there is no CV, so we use defaults (alpha = 1, min_df = 1) and document the choice.
- Best config per size is refit on the full subset and evaluated on test. Every variant's test score is also logged; these are reported but never used for selection.
- Metrics: accuracy, macro-F1, Brier score and ECE, as mean ± 95% CI across seeds. Per-item predictions are kept for McNemar.

## 6. Crossover
The smallest n where NB's mean accuracy ≥ Decider's and an exact McNemar test (each seed vs Decider, same items) is not significant at α = 0.05, or NB is significantly better. Reported per seed and as a majority over seeds. If NB never reaches Decider, we say so and report the best gap with its CI.

## 7. Energy measurement (orion)
- Pre-flight: `nvidia-smi` shows both GPUs idle, and llama-swap/vLLM are not holding a GPU. **If either is busy, stop and ask.**
- GPU: pynvml `nvmlDeviceGetTotalEnergyConsumption` on the device index Decider uses (pinned with `CUDA_VISIBLE_DEVICES`).
- CPU: RAPL `package-0` `energy_uj`, read through the read-only bind mount at `/host/powercap`. Wrap-around is handled with `max_energy_range_uj`. orion's Zen 2 CPU exposes no DRAM domain, which is noted as a limitation. If a counter can't be read, **stop and ask. Never estimate.**
- Idle baseline: 60 s before each measured phase. Marginal = gross − idle_power × duration. Both are reported.
- Phases measured separately: Decider load, Decider warmup (excluded), Decider inference (bs = 1 and best bs), NB training per curve point, NB inference.
- Wall power: if orion's PSU/wall Prometheus exporter is reachable, query the same windows and cross-check.
- Latency: per-item p50/p95 and total wall-clock time for both methods.
- NB runs single-process on CPU. GPU idle draw during NB runs is reported but not attributed to NB.
- Small NB fits (e.g. 1 example per class) finish in milliseconds, which is below a clean RAPL window. Each fit and predict is repeated in a loop until the window is at least 10 s, then divided by the number of repeats. The repeat count is reported.

## 8. Outputs
- `results/*.csv` (decider_runs, nb_runs, nb_variants, energy, latency, contamination, versions).
- `plots/learning_curve.png`: log-x axis, NB best config with CI bands, Decider as a horizontal line with its CI band.
- `plots/energy.png`: kWh per 1k predictions, gross and marginal.
- `REPORT.md`: dataset decision, crossover, energy table, break-even point (N predictions where NB_train + N·NB_infer < N·Decider_infer, i.e. N > NB_train / (Decider_infer − NB_infer)), threats to validity, all versions. It notes that Decider's training energy is out of scope.
- `run_all.sh`: runs end to end with resumable stages (checkpoints by output file).

## 9. Assumptions and open questions
1. **Dataset:** WOS-46985 level-1, approved 2026-10-05.
2. **Docker:** approved. Everything runs in the `nb-vs-decider` image (see Execution environment).
3. **Model version:** pin the newest release the official PyPI package loads (v21 if compatible, else v19). Decider ships often, so I'll freeze at implementation time.
4. **RAPL permissions on orion:** not needed. The read-only bind mount works (tested 2026-10-05).
5. Execution: implementation by Sonnet subagents per task, with the main agent reviewing and committing.
