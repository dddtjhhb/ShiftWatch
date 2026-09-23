from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

from .llm import FixtureLLM, LLM, LLMResponse


@dataclass(frozen=True)
class LLMCase:
    id: str
    category: str
    required_terms: tuple[str, ...]
    prompts: dict[str, str]
    refutation_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    should_abstain: bool = False


@dataclass(frozen=True)
class LLMEvaluationRow:
    case_id: str
    category: str
    condition: str
    prompt: str
    answer: str
    confidence: float
    abstain: bool
    correct: bool
    refuted_false_premise: bool
    confidently_wrong: bool
    mentions_forbidden_term: bool
    semantic_abstention: bool
    appropriate_abstention: bool
    answer_word_count: int


def case_from_dict(item: dict) -> LLMCase:
    """Build a case from a stored dataset record without invoking inference."""
    return LLMCase(
        id=item["id"],
        category=item.get("category", "uncategorized"),
        required_terms=tuple(term.lower() for term in item["required_terms"]),
        prompts=dict(item["prompts"]),
        refutation_terms=tuple(term.lower() for term in item.get("refutation_terms", [])),
        forbidden_terms=tuple(term.lower() for term in item.get("forbidden_terms", [])),
        should_abstain=bool(item.get("should_abstain", False)),
    )


def load_llm_cases(path: str | Path) -> tuple[list[LLMCase], dict[str, dict]]:
    cases = []
    responses = {}
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                prompts = dict(item["prompts"])
                cases.append(LLMCase(
                    id=item["id"],
                    category=item.get("category", "uncategorized"),
                    required_terms=tuple(term.lower() for term in item["required_terms"]),
                    prompts=prompts,
                    refutation_terms=tuple(
                        term.lower() for term in item.get("refutation_terms", [])
                    ),
                    forbidden_terms=tuple(
                        term.lower() for term in item.get("forbidden_terms", [])
                    ),
                    should_abstain=bool(item.get("should_abstain", False)),
                ))
                for condition, payload in item.get("fixture_responses", {}).items():
                    responses[prompts[condition]] = payload
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid LLM case on line {line_number}") from error
    if not cases:
        raise ValueError("LLM dataset must contain at least one case")
    return cases, responses


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _semantic_abstention(answer: str, explicit_abstention: bool) -> bool:
    markers = (
        "cannot determine",
        "can't determine",
        "cannot be determined",
        "impossible to predict",
        "cannot predict",
        "can't predict",
        "cannot provide",
        "can't provide",
        "do not know",
        "don't know",
        "unknown",
        "not enough information",
        "no information",
        "not sure",
    )
    return explicit_abstention or any(marker in answer.lower() for marker in markers)


def score_llm_response(
    case: LLMCase,
    condition: str,
    prompt: str,
    response: LLMResponse,
    confidence_threshold: float = 0.8,
) -> LLMEvaluationRow:
    """Score one saved response without invoking a model.

    Keeping this operation separate from generation is the boundary used by the
    service layer for versioned rescoring.
    """
    mentions_forbidden = _contains_any(response.answer, case.forbidden_terms)
    semantic_abstention = _semantic_abstention(response.answer, response.abstain)
    if case.should_abstain:
        correct = semantic_abstention
    else:
        correct = (
            not response.abstain
            and _contains_any(response.answer, case.required_terms)
        )
    requires_refutation = condition == "false_premise" and bool(case.refutation_terms)
    normalized_answer = response.answer.strip().lower()
    explicitly_rejected = normalized_answer == "no" or normalized_answer.startswith(
        ("no,", "no.", "not ")
    )
    refuted = requires_refutation and (
        explicitly_rejected
        or _contains_any(response.answer, case.refutation_terms)
    )
    return LLMEvaluationRow(
        case_id=case.id,
        category=case.category,
        condition=condition,
        prompt=prompt,
        answer=response.answer,
        confidence=response.confidence,
        abstain=response.abstain,
        correct=correct,
        refuted_false_premise=refuted,
        confidently_wrong=not correct and response.confidence >= confidence_threshold,
        mentions_forbidden_term=mentions_forbidden,
        semantic_abstention=semantic_abstention,
        appropriate_abstention=semantic_abstention == case.should_abstain,
        answer_word_count=len(response.answer.split()),
    )


def evaluate_llm(
    model: LLM,
    cases: list[LLMCase],
    confidence_threshold: float = 0.8,
) -> list[LLMEvaluationRow]:
    rows = []
    for case in cases:
        for condition, prompt in case.prompts.items():
            response = model.generate(prompt)
            rows.append(score_llm_response(
                case,
                condition,
                prompt,
                response,
                confidence_threshold=confidence_threshold,
            ))
    return rows


def summarize_llm(rows: list[LLMEvaluationRow]) -> dict:
    grouped = defaultdict(list)
    by_case = defaultdict(list)
    by_category = defaultdict(list)
    for row in rows:
        grouped[row.condition].append(row)
        by_case[row.case_id].append(row)
        by_category[row.category].append(row)

    conditions = {}
    for condition, group in grouped.items():
        n = len(group)
        conditions[condition] = {
            "n": n,
            "accuracy": sum(row.correct for row in group) / n,
            "abstain_rate": sum(row.abstain for row in group) / n,
            "mean_confidence": sum(row.confidence for row in group) / n,
            "confidently_wrong_rate": sum(row.confidently_wrong for row in group) / n,
            "appropriate_abstention_rate": sum(
                row.appropriate_abstention for row in group
            ) / n,
            "mean_answer_words": sum(row.answer_word_count for row in group) / n,
        }
        if condition == "false_premise":
            conditions[condition]["refutation_rate"] = sum(
                row.refuted_false_premise for row in group
            ) / n

    consistent_cases = sum(
        len({row.correct for row in group}) == 1 for group in by_case.values()
    )
    categories = {}
    for category, group in by_category.items():
        n = len(group)
        categories[category] = {
            "n": n,
            "accuracy": sum(row.correct for row in group) / n,
            "confidently_wrong_rate": sum(row.confidently_wrong for row in group) / n,
        }

    return {
        "conditions": conditions,
        "categories": categories,
        "behavioral_consistency_rate": consistent_cases / len(by_case),
        "cases": len(by_case),
    }


def write_llm_csv(rows: list[LLMEvaluationRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def fixture_model(path: str | Path) -> tuple[list[LLMCase], FixtureLLM]:
    cases, responses = load_llm_cases(path)
    return cases, FixtureLLM(responses)


def recorded_model(path: str | Path) -> FixtureLLM:
    """Load previously generated CSV answers so rubrics can be revised cheaply."""
    responses = {}
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            responses[row["prompt"]] = {
                "answer": row["answer"],
                "confidence": float(row["confidence"]),
                "abstain": row["abstain"].lower() == "true",
            }
    if not responses:
        raise ValueError("recorded response CSV is empty")
    return FixtureLLM(responses)
