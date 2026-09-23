"""Scoring runs: deterministic re-scoring of stored outputs. Never calls a model."""
from collections import defaultdict
from dataclasses import asdict

import psycopg
from psycopg.types.json import Jsonb

from ..llm import LLMResponse
from ..llm_evaluation import LLMEvaluationRow, case_from_dict, score_llm_response, summarize_llm
from .repository import Conflict, NotFound, canonical_sha256

RUBRICS = {"keyword_v1"}
OVERRIDABLE_FIELDS = {"required_terms", "refutation_terms", "forbidden_terms", "should_abstain"}
TERMINAL_RUN_STATUSES = ("succeeded", "partially_failed", "failed", "cancelled")


def _unparsed_row(case, condition, prompt) -> LLMEvaluationRow:
    """A reply that broke the JSON contract is scored as incorrect and non-abstaining."""
    return LLMEvaluationRow(
        case_id=case.id, category=case.category, condition=condition, prompt=prompt,
        answer="", confidence=0.0, abstain=False, correct=False,
        refuted_false_premise=False, confidently_wrong=False,
        mentions_forbidden_term=False, semantic_abstention=False,
        appropriate_abstention=not case.should_abstain, answer_word_count=0,
    )


def create_scoring_run(
    conn: psycopg.Connection,
    inference_run_id,
    rubric: str = "keyword_v1",
    confidence_threshold: float = 0.8,
    case_overrides: dict | None = None,
) -> tuple[dict, bool]:
    """Score every stored output of a finished run under a versioned rubric.

    ``case_overrides`` lets a rubric revision change per-case terms without creating
    a new dataset version or re-running inference. The same rubric+params on the same
    run is idempotent (returns the existing scoring run).
    """
    if rubric not in RUBRICS:
        raise ValueError(f"unknown rubric {rubric!r}")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be within [0, 1]")
    case_overrides = case_overrides or {}
    for case_id, fields in case_overrides.items():
        unknown = set(fields) - OVERRIDABLE_FIELDS
        if unknown:
            raise ValueError(f"case {case_id}: cannot override {sorted(unknown)}")
    params = {"confidence_threshold": confidence_threshold, "case_overrides": case_overrides}
    digest = canonical_sha256({"rubric": rubric, "params": params})

    with conn.transaction():
        run = conn.execute(
            "SELECT id, status, dataset_version_id FROM inference_runs WHERE id = %s",
            (inference_run_id,),
        ).fetchone()
        if run is None:
            raise NotFound(f"run {inference_run_id} not found")
        if run["status"] not in TERMINAL_RUN_STATUSES:
            raise Conflict(f"run is {run['status']}; score it after it finishes")
        existing = conn.execute(
            "SELECT * FROM scoring_runs WHERE inference_run_id = %s AND rubric_sha256 = %s",
            (inference_run_id, digest),
        ).fetchone()
        if existing:
            return existing, False

        cases = {}
        for row in conn.execute(
            "SELECT case_id, payload FROM dataset_cases WHERE dataset_version_id = %s",
            (run["dataset_version_id"],),
        ).fetchall():
            cases[row["case_id"]] = case_from_dict({**row["payload"], **case_overrides.get(row["case_id"], {})})
        unknown_cases = set(case_overrides) - set(cases)
        if unknown_cases:
            raise ValueError(f"overrides reference unknown cases {sorted(unknown_cases)}")

        tasks = conn.execute(
            "SELECT t.id, t.case_id, t.condition, t.prompt, o.answer, o.confidence,"
            " o.abstain, o.parse_error, o.task_id IS NOT NULL AS has_output"
            " FROM inference_tasks t LEFT JOIN model_outputs o ON o.task_id = t.id"
            " WHERE t.run_id = %s ORDER BY t.id",
            (inference_run_id,),
        ).fetchall()
        scored, missing = [], 0
        for task in tasks:
            if not task["has_output"]:
                missing += 1  # failed/cancelled: excluded from metrics, reported as coverage
                continue
            case = cases[task["case_id"]]
            if task["parse_error"]:
                row = _unparsed_row(case, task["condition"], task["prompt"])
            else:
                row = score_llm_response(
                    case, task["condition"], task["prompt"],
                    LLMResponse(task["answer"], task["confidence"], task["abstain"]),
                    confidence_threshold,
                )
            scored.append((task["id"], row, bool(task["parse_error"])))

        scoring = conn.execute(
            "INSERT INTO scoring_runs (inference_run_id, rubric, rubric_sha256, params,"
            " scored_outputs, missing_outputs) VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
            (inference_run_id, rubric, digest, Jsonb(params), len(scored), missing),
        ).fetchone()
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO scores (scoring_run_id, task_id, case_id, category, condition,"
                " correct, details) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [
                    (scoring["id"], task_id, row.case_id, row.category, row.condition,
                     row.correct, Jsonb({**asdict(row), "parse_error": parse_error}))
                    for task_id, row, parse_error in scored
                ],
            )
    return scoring, True


def scoring_summary(conn: psycopg.Connection, scoring_run_id) -> dict:
    scoring = conn.execute(
        "SELECT * FROM scoring_runs WHERE id = %s", (scoring_run_id,)
    ).fetchone()
    if scoring is None:
        raise NotFound(f"scoring run {scoring_run_id} not found")
    fields = LLMEvaluationRow.__dataclass_fields__
    rows = [
        LLMEvaluationRow(**{k: v for k, v in r["details"].items() if k in fields})
        for r in conn.execute(
            "SELECT details FROM scores WHERE scoring_run_id = %s ORDER BY task_id",
            (scoring_run_id,),
        ).fetchall()
    ]
    summary = summarize_llm(rows) if rows else {"conditions": {}, "categories": {}, "cases": 0}
    summary["parse_errors"] = conn.execute(
        "SELECT count(*) AS n FROM scores WHERE scoring_run_id = %s"
        " AND (details->>'parse_error')::boolean",
        (scoring_run_id,),
    ).fetchone()["n"]
    return {"scoring_run": scoring, "summary": summary}


def compare_scoring_runs(conn: psycopg.Connection, scoring_run_ids: list) -> dict:
    """Accuracy by condition, side by side, computed in SQL (uses scores_condition_idx)."""
    table = defaultdict(dict)
    for row in conn.execute(
        "SELECT scoring_run_id, condition, count(*) AS n,"
        " avg(correct::int)::float AS accuracy"
        " FROM scores WHERE scoring_run_id = ANY(%s)"
        " GROUP BY scoring_run_id, condition ORDER BY condition",
        (scoring_run_ids,),
    ).fetchall():
        table[row["condition"]][str(row["scoring_run_id"])] = {
            "n": row["n"], "accuracy": row["accuracy"]
        }
    return {"by_condition": table}
