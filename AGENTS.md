# AGENTS.md — Naive Bayes vs AWS Strands Decider 2B

Benchmark: how many labelled examples does scikit-learn Naive Bayes need to match the
zero-shot accuracy of AWS Strands Decider 2B on one text-classification dataset, and what
does each approach cost in energy (kWh)? Full plan: [`docs/PLAN.md`](docs/PLAN.md).

## Headline outputs
1. **Crossover point** — smallest examples-per-class where NB mean accuracy ≥ Decider's and a
   McNemar test on the same test items is not significant (or NB is significantly better).
2. **Energy** — kWh per 1,000 predictions for each method, plus NB training kWh at every
   learning-curve point (and at the crossover).

## Layout
```
docs/PLAN.md          approved plan, dataset decision log, open questions
docs/PLAN_STABILITY.md  follow-up: determinism/flip-rate test (runs after main REPORT.md exists)
  (stability deliverables go in nb-vs-decider/stability/)
nb-vs-decider/        ALL deliverables live here
  scripts/            pipeline stages (one module per stage)
  Dockerfile, docker-compose.yml, preflight_host.sh
  run_all.sh          one-command end-to-end entry point
  config/             pinned versions, label descriptions (versioned), seeds
  results/*.csv       raw per-run metrics (never hand-edited)
  plots/              learning_curve.png, energy.png
  REPORT.md           final report
```

## Hard rules (from the project owner)
- **GPU safety:** run measurements on host **orion** (2× RTX 3090). Before any energy
  measurement, check `nvidia-smi`. If vLLM or llama-swap holds the GPU, **STOP and ask**.
  Never stop, kill or restart those services.
- **Energy from hardware counters only:** NVML `nvmlDeviceGetTotalEnergyConsumption` (pynvml)
  on the GPU actually used; CPU from RAPL (`/sys/class/powercap/intel-rapl*`) or AMD equivalent.
  If counters are unreadable, **ask for permissions — never fall back to TDP estimates.**
- Measure a 60 s idle baseline; report gross **and** marginal (gross − idle power × duration).
  Exclude warmup/model loading from per-prediction energy; report loading separately.
  Decider's CPU energy (tokenisation, host work) counts. Decider training energy is out of scope.
- **Never tune on the test set.** Test split is fixed, stratified, identical for both methods.
  Dev split is used for NB CV/selection and Decider label wording only.
- **Decider label descriptions** are written once before any test result is seen. Iterate only
  on dev, only by rewording (never by adding examples). Log every version in `config/`.
- **NB:** scikit-learn only (`MultinomialNB`, `ComplementNB` × `CountVectorizer`,
  `TfidfVectorizer`, ngram (1,2)); tune `alpha`, `min_df` by stratified CV on the training subset.
- If a number looks surprising, re-run with another seed before reporting it.
- Plan first; show the plan before running anything long.

## Environment & tooling
- **Everything runs in Docker** (`nb-vs-decider` image, compose services `pipeline` (GPU 0) and `dev` (CPU)).
  Only `preflight_host.sh` (GPU-idle / llama-swap / vLLM check) runs on the host, via `run_all.sh`.
- Python inside the image: `uv` venv at `/opt/venv`, pinned via `uv.lock`; export `requirements.txt` and
  `requirements-dev.txt`. Build multi-arch (`linux/amd64`, `linux/arm64`) with buildx.
- RAPL in containers: Docker masks `/sys/devices/virtual/powercap`; bind-mount it read-only to
  `/host/powercap`. Do NOT use `privileged` or disable AppArmor.
- Record package, driver, CUDA, CPU model and Decider versions in the report.
- `ruff` for lint/format (flake8 + black + isort rules). Standard `logging` module only.
- Config via environment variables loaded with `python-dotenv` (`.env`, never committed).
- Determinism: every random choice takes an explicit seed; seeds are listed in config.
- Pin Decider: PyPI `strands-decider` version + Hugging Face model revision SHA + GitHub commit.

## Workflow
- Commit after each task; push only when asked.
- Write a journal entry (journal skill) after each meaningful piece of work.
