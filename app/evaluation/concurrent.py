from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from app.config import settings
from app.errors import RunError, find_run_error

CaseFn = Callable[[dict], dict]


def resolve_concurrency(cli: int | None = None) -> int:
    if cli is not None:
        return max(1, int(cli))
    return max(1, int(getattr(settings, "eval_concurrency", 8) or 8))


def failed_record(case: dict, exc: BaseException) -> dict:
    cid = case.get("id")
    err = find_run_error(exc)
    if err is None:
        err = RunError(str(exc)[:800])
    rec = err.to_record(cid)
    rec["error"] = rec.get("error") or str(exc)[:800]
    return rec


def _guarded(fn: CaseFn) -> CaseFn:
    def wrap(case: dict) -> dict:
        try:
            out = fn(case)
            if not isinstance(out, dict):
                cid = case.get("id")
                return {"id": cid, "question_id": cid, "status": "failed", "error": "case worker returned non-dict"}
            out.setdefault("id", case.get("id"))
            return out
        except Exception as exc:
            return failed_record(case, exc)

    return wrap


async def _map_async(cases: Sequence[dict], fn: CaseFn, concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(max(1, concurrency))
    worker = _guarded(fn)

    async def one(case: dict) -> dict:
        async with sem:
            return await asyncio.to_thread(worker, case)

    return list(await asyncio.gather(*(one(c) for c in cases)))


def run_cases(cases: Sequence[dict], fn: CaseFn, concurrency: int) -> list[dict]:
    """按 dataset 顺序返回；单 case 失败记 failed，不抛出。"""
    n = max(1, int(concurrency))
    return asyncio.run(_map_async(cases, fn, n))
