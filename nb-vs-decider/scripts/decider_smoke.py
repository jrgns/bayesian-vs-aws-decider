"""Load Decider via the official reference code and reproduce the model-card example."""

from __future__ import annotations

import logging
import statistics
import time

from scripts.common import load_settings, prepare_checkpoint, setup_logging, write_json

log = logging.getLogger(__name__)

TOLERANCE = 0.005
TIMING_WARMUP = 3
TIMING_CALLS = 50
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


# Short states written for the batched-path consistency check (no benchmark data).
BATCH_STATES = [
    "My card was charged twice for the same invoice.",
    "I would like a quote for 200 seats on the enterprise plan.",
    "The store never shipped my order and I want my money back.",
    "Refund still not showing up on my statement after two weeks.",
    "Do you offer a discount if we sign a three year contract?",
    "The checkout page keeps crashing when I try to pay.",
    "Where is the nearest shop to collect my parcel?",
    "Please update the bank account used for my monthly payouts.",
]
CHOICE_INSTRUCTION = "Which team should handle this?"
CHOICE_OPTIONS = ["billing", "sales", "retail"]


def max_abs_diff(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def batched_check(model, engine, schema) -> dict:
    """Compare engine.ask per state with collect_logits at bs=1 and bs=8, same temperature."""
    import torch
    from strands_decider.data.format import Example
    from strands_decider.evaluate import collect_logits, predictions_from_logits

    examples = [
        Example(
            kind="choice",
            state=s,
            instructions=CHOICE_INSTRUCTION,
            options=[[o, ""] for o in CHOICE_OPTIONS],
            label=0,
        )
        for s in BATCH_STATES
    ]
    question = {
        "q": schema.ChoiceQuestion(
            instructions=CHOICE_INSTRUCTION, criteria={o: "" for o in CHOICE_OPTIONS}
        )
    }
    ask_probs = []
    for s in BATCH_STATES:
        ans = engine.ask(s, question).answers["q"]
        ask_probs.append([ans.probabilities[o] for o in CHOICE_OPTIONS])

    temps = model.config.temperature_by_kind
    max_len = model.config.max_length
    logits = {}
    probs = {}
    for bs in (1, 8):
        lg, labels, n_slots, exs = collect_logits(
            model, examples, device="cuda", batch_size=bs, max_length=max_len
        )
        logits[bs] = lg
        preds = predictions_from_logits(lg, labels, n_slots, exs, temperature=temps)
        probs[bs] = [p.probs for p in preds]

    argmax = lambda rows: [max(range(len(r)), key=r.__getitem__) for r in rows]  # noqa: E731
    return {
        "n_states": len(examples),
        "argmax_agree_ask_vs_bs1": argmax(ask_probs) == argmax(probs[1]),
        "argmax_agree_ask_vs_bs8": argmax(ask_probs) == argmax(probs[8]),
        "argmax_agree_bs1_vs_bs8": argmax(probs[1]) == argmax(probs[8]),
        "max_abs_prob_diff_ask_vs_bs1": max(
            max_abs_diff(a, b) for a, b in zip(ask_probs, probs[1], strict=True)
        ),
        "max_abs_prob_diff_ask_vs_bs8": max(
            max_abs_diff(a, b) for a, b in zip(ask_probs, probs[8], strict=True)
        ),
        "max_abs_prob_diff_bs1_vs_bs8": max(
            max_abs_diff(a, b) for a, b in zip(probs[1], probs[8], strict=True)
        ),
        "logits_bitwise_identical_bs1_vs_bs8": bool(torch.equal(logits[1], logits[8])),
        "max_abs_logit_diff_bs1_vs_bs8": float((logits[1] - logits[8]).abs().max()),
        "temperature_applied": dict(temps),
        "how_to_use_collect_logits": (
            "from strands_decider.data.format import Example; "
            "from strands_decider.evaluate import collect_logits, predictions_from_logits. "
            "Build Example(kind='choice', state=<text>, instructions=<question>, "
            "options=[[name, description], ...], label=<int index into options>); "
            "description may be ''. Prompt text is rendered by the package itself "
            "(Example.to_question -> prompting.render_state/render_question), so no manual "
            "formatting. Call collect_logits(model, examples, device='cuda', batch_size=N, "
            "max_length=model.config.max_length): the default max_length=3072 is NOT the "
            "model's 4096 window, so always pass it. It returns raw untempered logits "
            "(columns past each row's option count hold -1e4), labels, n_slots, examples; "
            "options are never shuffled in eval. Apply temperature with "
            "predictions_from_logits(logits, labels, n_slots, examples, "
            "temperature=model.config.temperature_by_kind): divides logits by the per-kind "
            "temperature (choice=0.8177 for v21) then masked softmax. "
            "predictions_from_logits returns Prediction objects with .probs, .pred. "
            "collect_logits uses the model on its current device but does not use the "
            "engine's prefix cache or fp32 handling (cuda keeps bf16 torso, fp32 head)."
        ),
    }


def time_ask(engine, state, questions) -> dict:
    for _ in range(TIMING_WARMUP):
        engine.ask(state, questions)
    times = []
    for _ in range(TIMING_CALLS):
        t0 = time.perf_counter()
        engine.ask(state, questions)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "calls": TIMING_CALLS,
        "median_ms": statistics.median(times),
        "p95_ms": sorted(times)[int(0.95 * len(times)) - 1],
        "min_ms": min(times),
    }


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

    timing = time_ask(engine, STATE, questions)
    log.info("ask latency (3-question card example): %s", timing)
    batched = batched_check(model, engine, schema)
    log.info(
        "batched check: %s", {k: v for k, v in batched.items() if k != "how_to_use_collect_logits"}
    )

    cfg = model.config
    write_json(
        "decider_smoke.json",
        {
            "all_pass": all(c["pass"] for c in checks.values()),
            "tolerance": TOLERANCE,
            "checks": checks,
            "usage": resp.usage.model_dump(),
            "ask_latency": timing,
            "batched_path": batched,
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
