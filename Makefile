PY        ?= python
CONFIG    ?= configs/synthetic.yaml
ARTIFACTS ?= artifacts/synthetic
# Development keys for `make serve` only; never reuse them anywhere else.
DEV_SCORER_KEY  ?= dev-scorer-key-change-me
DEV_ANALYST_KEY ?= dev-analyst-key-change-me

.PHONY: install train train-ieee serve replay test lint docker-build docker-run

install:            ## install the package with dev tools
	$(PY) -m pip install -e ".[dev]"

train:              ## full pipeline on synthetic IEEE-CIS-shaped data (~2 min)
	$(PY) -m fraud_detection.train --config configs/synthetic.yaml

train-ieee:         ## full pipeline on the real Kaggle data (see data/README.md)
	$(PY) -m fraud_detection.train --config configs/ieee.yaml

serve:              ## scoring API on http://127.0.0.1:8000 (docs at /docs), dev keys
	FRAUD_ARTIFACTS=$(ARTIFACTS) FRAUD_API_KEYS=$(DEV_SCORER_KEY) \
	FRAUD_ANALYST_API_KEYS=$(DEV_ANALYST_KEY) \
	uvicorn fraud_detection.api.app:app --host 127.0.0.1 --port 8000

replay:             ## stream the test month through an in-process API: skew, latency, drift
	$(PY) -m fraud_detection.replay --config $(CONFIG) --limit 5000

test:
	$(PY) -m pytest

lint:
	ruff check src tests && ruff format --check src tests

docker-build:       ## image with the trained bundle baked in (run `make train` first)
	docker build --build-arg ARTIFACTS=$(ARTIFACTS) -t fraud-api .

docker-run:         ## needs FRAUD_API_KEYS / FRAUD_ANALYST_API_KEYS in your environment
	docker run --rm -p 8000:8000 -e FRAUD_API_KEYS -e FRAUD_ANALYST_API_KEYS fraud-api
