"""Validate and retain DiscoveryBench native evaluation results."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time

EVALUATION_PROTOCOL = "discoverybench-native-v1"


def input_digest(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()


def validate_score(record):
    score = record.get("final_score")
    if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"Invalid native final_score: {score}")
    # Native parsing uses -1 sentinels; their products must not masquerade as valid scores.
    for sub in record.get("matched_gold_gen_subh_evals", {}).values():
        for value in (sub["context"]["score"], sub["var"]["score"]["f1"], sub["rel"]["score"]):
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Judge returned an invalid context/variable/relation score")
    for field in ("gold_sub_hypo", "gen_sub_hypo"):
        if any(sub.get("explanation") == "unable to segment" for sub in record.get(field, {}).get("sub_hypo", [])):
            # Preserve the native fallback and its computed score in the task
            # record, but distinguish unresolved extraction from a scored task.
            if field == "gen_sub_hypo" and score == 0:
                continue
            raise ValueError("Judge failed to extract subhypotheses; retry this evaluation")
    return float(score)


def accepted_score(record, job=None):
    """Resolve only records produced by the original evaluator for these inputs."""
    if record.get("evaluation_protocol") != EVALUATION_PROTOCOL or record.get("status") != "scored":
        return None
    digest = record.get("evaluation_inputs_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        return None
    if job is not None and record.get("evaluation_inputs_sha256") != input_digest(job):
        return None
    native = record.get("native_evaluation")
    if not isinstance(native, dict) or record.get("score") != native.get("final_score"):
        return None
    try:
        return validate_score(native)
    except (KeyError, TypeError, ValueError):
        return None


def evaluate_job(job):
    result = {key: job[key] for key in ("task_id", "split", "dataset", "metadata_id", "query_id", "generation_status")}
    result.update(evaluation_protocol=EVALUATION_PROTOCOL, evaluation_inputs_sha256=input_digest(job))
    hypothesis = job["arguments"].get("gen_hypo")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return dict(result, status="generation_failure", score=None, reason="No nonempty hypothesis for the native evaluator")
    started = time.monotonic()
    try:
        from .discoverybench_transport import configure_judge
        configure_judge()
        from eval.new_eval import run_eval_gold_vs_gen_NL_hypo_workflow
        native = run_eval_gold_vs_gen_NL_hypo_workflow(**job["arguments"], llm_used=job["judge_model"])
        result.update(native_evaluation=native, score=native.get("final_score"))
        validate_score(native)
        result["status"] = "scored"
    except Exception as error:
        result.setdefault("score", None)
        result.update(status="evaluation_error", error=f"{type(error).__name__}: {error}")
    result["runtime"] = time.monotonic() - started
    return result
