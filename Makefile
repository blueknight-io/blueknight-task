.PHONY: install ingest run test lint

PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
VENV_UVICORN := $(VENV_DIR)/bin/uvicorn

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV_DIR)

install: $(VENV_PYTHON)
	$(VENV_PIP) install -r requirements.txt

ingest: $(VENV_PYTHON)
	$(VENV_PYTHON) scripts/ingest.py

run: $(VENV_PYTHON)
	$(VENV_UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000

test: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pytest tests/ -v --asyncio-mode=auto

lint: $(VENV_PYTHON)
	$(VENV_PYTHON) -m py_compile \
		app/config.py \
		app/schemas.py \
		app/retrieval.py \
		app/utils/json_contract.py \
		app/utils/logging.py \
		app/utils/normalization.py \
		app/services/vector_store.py \
		app/services/retrieval_wrapper.py \
		app/services/post_filter.py \
		app/services/reranker.py \
		app/services/search_pipeline.py \
		app/services/refiner.py \
		app/main.py \
		scripts/ingest.py
