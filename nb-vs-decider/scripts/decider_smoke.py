"""Load Decider via the official reference code and reproduce the model-card example."""

from __future__ import annotations

import logging
import time

from scripts.common import load_settings, prepare_checkpoint, setup_logging, write_json

log = logging.getLogger(__name__)

TOLERANCE = 0.005
STATE = "Help! My payouts have been failing for 3 days! "

# Published v21 output on cuda (README / model card of strands-decider-2B-hobson-v21).
EXPECTED = {
    "noul": 0.875,
    "choice.billing": 0.891,
    "choice.sales": 0.055,
    "choice.retail": 0.054,
    "choice.confidence": 0.837,
    "score": 1.07,
    "score.calm": 0.140,
    "score.frustrated": 0.649,
    "score.depressed": 0.211,
}


def compare(expected: dict[str, float], actual: dict[str, float], tol: float) -> dict[str, dict]:
    """Per-key pass/fail: |actual - expected| <= tol. A missing actual value fails."""
    out = {}
    for key, exp in expected.items():
        act = actual.get(key)
        diff = None if act is None else abs(act - exp)
        out[key] = {
            "expected": exp,
            "actual": act,
            "abs_diff": diff,
            "pass": diff is not None and diff <= tol,
        }
    return out


def flatten(answers) -> dict[str, float]:
    """Flatten the engine response into the keys used by EXPECTED."""
    c, s = answers["choice"], answers["score"]
    flat = {"noul": answers["noul"].noul, "choice.confidence": c.confidence, "score": s.score}
    flat.update({f"choice.{k}": v for k, v in c.probabilities.items()})
    flat.update({f"score.{s.legend[k]}": v for k, v in s.probabilities.items()})
    return flat


def main() -> None:
    s = load_settings()
    setup_logging(s.log_level)
    import torch
    from strands_decider import schema
    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.modeling import StrandsDeciderModel

    ckpt = prepare_checkpoint(s)
    t0 = time.perf_counter()
    model = StrandsDeciderModel.load(ckpt.path)
    engine = SystemOneEngine(model, EngineConfig(device="cuda"))
    load_s = time.perf_counter() - t0
    log.info("loaded %s@%s in %.1fs", ckpt.model_id, ckpt.model_revision, load_s)

    questions = {
        "noul": schema.NoulQuestion(instructions="Does this convey urgency?"),
        "choice": schema.ChoiceQuestion(
            instructions="Which team should handle this?",
            criteria={"billing": "", "sales": "", "retail": ""},
        ),
        "score": schema.ScoreQuestion(
            instructions="How frustrated is the writer?",
            criteria=["calm", "frustrated", "depressed"],
        ),
    }
    resp = engine.ask(STATE, questions)
    actual = flatten(resp.answers)
    checks = compare(EXPECTED, actual, TOLERANCE)
    for key, c in checks.items():
        log.info(
            "%-20s expected %.3f actual %.3f %s",
            key,
            c["expected"],
            c["actual"],
            "ok" if c["pass"] else "FAIL",
        )

    cfg = model.config
    write_json(
        "decider_smoke.json",
        {
            "all_pass": all(c["pass"] for c in checks.values()),
            "tolerance": TOLERANCE,
            "checks": checks,
            "usage": resp.usage.model_dump(),
            "load_seconds": load_s,
            "model": {"id": ckpt.model_id, "revision": ckpt.model_revision},
            "base": {"id": ckpt.base_model, "revision": ckpt.base_revision},
            "device": torch.cuda.get_device_name(0),
            "findings": {
                "max_context_tokens": cfg.max_length,
                "context_note": "state is truncated to fit; question keeps first claim",
                "head_type": cfg.head_type,
                "max_choice_options": schema.MAX_CHOICE_OPTIONS,
                "score_levels": [schema.MIN_SCORE_LEVELS, schema.MAX_SCORE_LEVELS],
                "num_slots_ignored_for_pointer_head": cfg.num_slots,
                "batching": (
                    "engine.ask(state, {name: Question}) takes ONE state and N questions; "
                    f"questions are batched (max_batch={engine.cfg.max_batch}) and share "
                    "the state's KV cache (use_prefix_cache). No API for several states per call: "
                    "multi-state batching needs custom code or one call per state."
                ),
                "probabilities": (
                    "ChoiceAnswer.probabilities: dict option -> probability (softmax, temperature "
                    "calibrated), plus .choice (argmax) and .confidence; ScoreAnswer.probabilities "
                    "keyed by level index with .legend; NoulAnswer.noul = P(true)."
                ),
            },
        },
    )


if __name__ == "__main__":
    main()
