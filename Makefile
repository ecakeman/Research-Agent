PYTHON ?= python3
PIP_INDEX ?= https://pypi.tuna.tsinghua.edu.cn/simple
COMPOSE ?= docker compose
BASELINE ?= agentic
MODEL_ROUTING ?= single
CONCURRENCY ?=

.PHONY: install db-up db-down migrate ingest test eval eval-live serve golden chunk-stats

install:
	$(PYTHON) -m pip install -i $(PIP_INDEX) -e ".[dev]"

db-up:
	$(COMPOSE) up -d postgres

db-down:
	$(COMPOSE) down

migrate:
	$(PYTHON) -m app.migrate

ingest:
	$(PYTHON) -m app.ingest

test:
	$(PYTHON) -m pytest -q

EVAL_CONC_FLAG = $(if $(CONCURRENCY),--concurrency $(CONCURRENCY),)

eval:
	$(PYTHON) -m app.evaluation.runner --baseline $(BASELINE) --routing $(MODEL_ROUTING) $(EVAL_CONC_FLAG)

eval-live:
	$(PYTHON) -m app.evaluation.runner --baseline $(BASELINE) --routing $(MODEL_ROUTING) --live $(EVAL_CONC_FLAG)

golden:
	PYTHONPATH=. $(PYTHON) tests/golden/runner.py

chunk-stats:
	$(PYTHON) -m app.evaluation.chunk_stats

serve:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8090
