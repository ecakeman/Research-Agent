from __future__ import annotations

import argparse
from datetime import datetime, timezone
from statistics import mean

from app.evaluation.concurrent import resolve_concurrency, run_cases
from app.evaluation.dataset import load_cases
from app.evaluation.metrics import mrr, ndcg_at_k, recall_at_k

NOT_READY = "Evaluation dataset not ready"


def _avg(xs: list[float]) -> float | None:
    return mean(xs) if xs else None


def _rewrite_placeholder() -> dict:
    return {
        "first_pass_insufficient": None,
        "eligible_for_recovery": None,
        "rewrite_attempted": None,
        "rewrite_recovered": None,
        "rewrite_recovery_rate": None,
    }


def _stamp(start: datetime, concurrency: int) -> dict:
    end = datetime.now(timezone.utc)
    return {
        "concurrency": concurrency,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_s": (end - start).total_seconds(),
    }


def run_eval(
    *,
    baseline: str = "agentic",
    live: bool = False,
    print_report: bool = True,
    model_routing: str = "single",
    concurrency: int | None = None,
) -> dict:
    conc = resolve_concurrency(concurrency)
    cases = load_cases()
    empty = {
        "error": NOT_READY,
        "cases": 0,
        "baseline": baseline,
        "model_routing": model_routing,
        "live": live,
        "failed": 0,
        "recall_5": None,
        "recall_10": None,
        "mrr": None,
        "ndcg_10": None,
        "groundedness": None,
        "citation_precision": None,
        "citation_recall": None,
        "correct_abstention": None,
        "abstention_rate": None,
        "failure_rate": None,
        "concurrency": conc,
        "start_time": None,
        "end_time": None,
        "duration_s": None,
        **_rewrite_placeholder(),
    }
    if not cases:
        if print_report:
            print(NOT_READY)
        return empty

    started = datetime.now(timezone.utc)
    if live:
        report = _live_eval(cases, baseline, model_routing, conc)
    else:
        report = _retrieval_eval(cases, baseline, model_routing, conc)
    report.update(_stamp(started, conc))
    if print_report:
        print_eval_report(report)
    return report


def _ranked_for_case(case: dict, baseline: str) -> list[str]:
    if case.get("ranked_ids"):
        return [str(x) for x in case["ranked_ids"]]
    relevant = [str(x) for x in (case.get("expected_sources") or []) if str(x).strip()]
    if not relevant:
        return []
    from app.evaluation.retrieve_eval import ranked_sources_for_query

    return ranked_sources_for_query(case["question"], baseline)


def _retrieval_one(case: dict, baseline: str) -> dict:
    relevant = {str(x) for x in (case.get("expected_sources") or []) if str(x).strip()}
    ranked = _ranked_for_case(case, baseline)
    out: dict = {"id": case.get("id"), "status": "completed"}
    if relevant:
        out["recall_5"] = recall_at_k(relevant, ranked, 5)
        out["recall_10"] = recall_at_k(relevant, ranked, 10)
        out["mrr"] = mrr(relevant, ranked)
        out["ndcg_10"] = ndcg_at_k(relevant, ranked, 10)
    expected_abs = case.get("category") in {"insufficient", "insufficient-evidence"}
    out["expected_abs"] = expected_abs
    if expected_abs:
        if "abstention_actual" in case:
            out["abstention_actual"] = bool(case.get("abstention_actual"))
            if out["abstention_actual"]:
                out["status"] = "abstained"
    elif case.get("status") == "abstained":
        out["status"] = "abstained"
    return out


def _retrieval_eval(cases: list[dict], baseline: str, model_routing: str, concurrency: int) -> dict:
    rows = run_cases(cases, lambda c: _retrieval_one(c, baseline), concurrency)
    r5, r10, mrrs, ndcgs = [], [], [], []
    abstain_exp = abstain_ok = 0
    completed = abstained = failed = 0
    labeled_abs = any(
        "abstention_actual" in c
        for c in cases
        if c.get("category") in {"insufficient", "insufficient-evidence"}
    )
    for case, row in zip(cases, rows, strict=True):
        if row.get("status") == "failed":
            failed += 1
            continue
        if row.get("recall_5") is not None:
            r5.append(row["recall_5"])
            r10.append(row["recall_10"])
            mrrs.append(row["mrr"])
            ndcgs.append(row["ndcg_10"])
        expected_abs = case.get("category") in {"insufficient", "insufficient-evidence"}
        if expected_abs:
            abstain_exp += 1
            if row.get("abstention_actual"):
                abstain_ok += 1
                abstained += 1
        elif row.get("status") == "abstained":
            abstained += 1
        else:
            completed += 1
    note = None
    if baseline == "agentic":
        note = "retrieval-only (same as hybrid+rerank); generation metrics Stage B"
    return {
        "baseline": baseline,
        "model_routing": model_routing,
        "live": False,
        "note": note,
        "cases": len(cases),
        "completed": completed,
        "abstained": abstained,
        "failed": failed,
        "case_results": rows,
        "recall_5": _avg(r5),
        "recall_10": _avg(r10),
        "mrr": _avg(mrrs),
        "ndcg_10": _avg(ndcgs),
        "groundedness": None,
        "citation_precision": None,
        "citation_recall": None,
        "correct_abstention": (abstain_ok / abstain_exp) if (abstain_exp and labeled_abs) else None,
        **_rewrite_placeholder(),
    }


def _live_one(case: dict, baseline: str) -> dict:
    from app.evaluation.live import score_live_case
    from app.service import execute_research

    result = execute_research(case["question"], baseline=baseline)
    scored = score_live_case(case, result)
    scored["id"] = case.get("id")
    return scored


def _live_eval(cases: list[dict], baseline: str, model_routing: str, concurrency: int) -> dict:
    from app.config import settings

    prev = settings.model_routing
    settings.model_routing = model_routing
    try:
        rows = run_cases(cases, lambda c: _live_one(c, baseline), concurrency)
    finally:
        settings.model_routing = prev
    return summarize_live(rows, baseline, model_routing)


def summarize_live(rows: list[dict], baseline: str, model_routing: str) -> dict:
    r5, r10, mrrs, ndcgs = [], [], [], []
    gnds, cps, crs = [], [], []
    latencies, tokens, pro_calls, fast_calls = [], [], [], []
    completed = abstained = failed = 0
    abstain_exp = abstain_ok = 0
    first_insuf = rewrite_n = recovered = eligible_n = 0
    for row in rows:
        status = row.get("status")
        if status == "completed":
            completed += 1
        elif status == "abstained":
            abstained += 1
        else:
            failed += 1
            status = "failed"
        if status != "failed" and row.get("recall_5") is not None:
            r5.append(row["recall_5"])
            r10.append(row["recall_10"])
            mrrs.append(row["mrr"])
            ndcgs.append(row["ndcg_10"])
        if status == "completed":
            if row.get("groundedness") is not None:
                gnds.append(row["groundedness"])
            if row.get("citation_precision") is not None:
                cps.append(row["citation_precision"])
            if row.get("citation_recall") is not None:
                crs.append(row["citation_recall"])
        if status in {"completed", "abstained"}:
            if row.get("abstention_expected"):
                abstain_exp += 1
                if row.get("abstention_actual"):
                    abstain_ok += 1
            latencies.append(row.get("latency_ms") or 0)
            tokens.append(row.get("total_tokens") or 0)
            pro_calls.append(row.get("pro_calls") or 0)
            fast_calls.append(row.get("fast_calls") or 0)
        if row.get("first_pass_insufficient"):
            first_insuf += 1
        if row.get("eligible_for_recovery"):
            eligible_n += 1
        if row.get("rewrite_attempted"):
            rewrite_n += 1
        if row.get("rewrite_recovered"):
            recovered += 1
    n = len(rows)
    rec_rate = (recovered / eligible_n) if eligible_n else None
    return {
        "baseline": baseline,
        "model_routing": model_routing,
        "live": True,
        "cases": n,
        "total": n,
        "completed": completed,
        "abstained": abstained,
        "failed": failed,
        "case_results": rows,
        "recall_5": _avg(r5),
        "recall_10": _avg(r10),
        "mrr": _avg(mrrs),
        "ndcg_10": _avg(ndcgs),
        "groundedness": _avg(gnds),
        "citation_precision": _avg(cps),
        "citation_recall": _avg(crs),
        "citation_precision_completed": _avg(cps),
        "citation_recall_completed": _avg(crs),
        "correct_abstention": (abstain_ok / abstain_exp) if abstain_exp else None,
        "abstention_rate": (abstained / n) if n else None,
        "failure_rate": (failed / n) if n else None,
        "first_pass_insufficient": first_insuf,
        "eligible_for_recovery": eligible_n,
        "rewrite_attempted": rewrite_n,
        "rewrite_recovered": recovered,
        "rewrite_recovery_rate": rec_rate,
        "avg_latency_ms": _avg(latencies),
        "total_tokens": sum(tokens),
        "pro_calls": sum(pro_calls),
        "fast_calls": sum(fast_calls),
    }


def print_eval_report(report: dict) -> None:
    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    print("Research Agent Evaluation")
    print("────────────────────────────────────")
    print(f"Baseline                    {report.get('baseline')}")
    print(f"Model routing               {report.get('model_routing')}")
    if report.get("note"):
        print(f"Note                        {report.get('note')}")
    print(f"Concurrency                 {fmt(report.get('concurrency'))}")
    print(f"Duration s                  {fmt(report.get('duration_s'))}")
    print(f"Cases                       {report.get('cases')}")
    print(f"Total                       {fmt(report.get('total'))}")
    print(f"Completed                   {fmt(report.get('completed'))}")
    print(f"Abstained                   {fmt(report.get('abstained'))}")
    print(f"Failed                      {fmt(report.get('failed'))}")
    print(f"Abstention rate             {fmt(report.get('abstention_rate'))}")
    print(f"Failure rate                {fmt(report.get('failure_rate'))}")
    print("")
    print("Retrieval (source-level)")
    print(f"Recall@5                    {fmt(report.get('recall_5'))}")
    print(f"Recall@10                   {fmt(report.get('recall_10'))}")
    print(f"MRR                         {fmt(report.get('mrr'))}")
    print(f"nDCG@10                     {fmt(report.get('ndcg_10'))}")
    print("")
    print("Completed-only")
    print(f"Groundedness                {fmt(report.get('groundedness'))}")
    print(f"Citation Precision          {fmt(report.get('citation_precision'))}")
    print(f"Citation Recall             {fmt(report.get('citation_recall'))}")
    print("")
    print("Insufficient Evidence")
    print(f"Correct Abstention          {fmt(report.get('correct_abstention'))}")
    print(f"first_pass_insufficient     {fmt(report.get('first_pass_insufficient'))}")
    print(f"eligible_for_recovery       {fmt(report.get('eligible_for_recovery'))}")
    print(f"rewrite_attempted           {fmt(report.get('rewrite_attempted'))}")
    print(f"rewrite_recovered           {fmt(report.get('rewrite_recovered'))}")
    print(f"rewrite_recovery_rate       {fmt(report.get('rewrite_recovery_rate'))}")
    if report.get("live"):
        print("")
        print("Cost")
        print(f"Avg Latency ms              {fmt(report.get('avg_latency_ms'))}")
        print(f"Total Tokens                {fmt(report.get('total_tokens'))}")
        print(f"Pro Calls                   {fmt(report.get('pro_calls'))}")
        print(f"Fast Calls                  {fmt(report.get('fast_calls'))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="agentic")
    parser.add_argument("--routing", default="single")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()
    run_eval(
        baseline=args.baseline,
        live=args.live,
        print_report=True,
        model_routing=args.routing,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
