from __future__ import annotations

import math


def recall_at_k(relevant: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = relevant.intersection(ranked_ids[:k])
    return len(hit) / len(relevant)


def mrr(relevant: set[str], ranked_ids: list[str]) -> float:
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: set[str], ranked_ids: list[str], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if cid in relevant else 0.0
        if rel:
            dcg += rel / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def citation_precision_recall(
    predicted: list[str], evidence_ids: set[str], expected: set[str] | None = None
) -> tuple[float, float]:
    pred = [p for p in predicted if p]
    if not pred:
        precision = 1.0 if not expected else 0.0
    else:
        valid = [p for p in pred if p in evidence_ids]
        precision = len(valid) / len(pred)
    gold = expected if expected is not None else evidence_ids
    if not gold:
        recall = 1.0
    else:
        recall = len(set(pred).intersection(gold)) / len(gold)
    return precision, recall
