"""Retrieval metrics and JSON-serializable per-query traces."""
import re
from typing import Any


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate automatic retrieval hits; manual quality labels remain separate."""
    gold_sources = set(question.get("gold_source_ids", [question["gold_source_id"]]))
    gold_pages = {int(page) for page in question.get("gold_pages", [question["gold_page"]])}
    evidence_quotes = [_normalize(quote) for quote in question.get(
        "evidence_quotes", [question["evidence_quote"]]
    )]

    def is_hit(hit: dict[str, Any]) -> bool:
        text = _normalize(str(hit.get("text", "")))
        return (
            hit.get("document_id") in gold_sources
            and int(hit.get("page", -1)) in gold_pages
            and any(quote in text for quote in evidence_quotes)
        )

    hit_flags = [is_hit(hit) for hit in hits]
    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "hit_at_1": bool(hit_flags[:1] and hit_flags[0]),
        "evidence_hit_at_3": any(hit_flags[:3]),
        "hits": hits,
        "gold": {
            "source_ids": sorted(gold_sources),
            "pages": sorted(gold_pages),
            "evidence_quotes": question.get("evidence_quotes", [question["evidence_quote"]]),
        },
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for trace in traces:
        strategy = trace["strategy"]
        split = trace["split"]
        bucket = summary.setdefault(strategy, {}).setdefault(
            split, {"total": 0, "hit_at_1": 0, "evidence_hit_at_3": 0}
        )
        bucket["total"] += 1
        bucket["hit_at_1"] += int(trace["hit_at_1"])
        bucket["evidence_hit_at_3"] += int(trace["evidence_hit_at_3"])
    return summary
