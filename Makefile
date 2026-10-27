PY        ?= python
CONFIG    ?= configs/synthetic.yaml
ARTIFACTS ?= artifacts/synthetic

.PHONY: install train train-ieee serve replay test lint docker-build docker-run

install:            ## install the package with dev tools
	$(PY) -m pip install -e ".[dev]"

train:              ## full pipeline on synthetic IEEE-CIS-shaped data (~2 min)
	$(PY) -m fraud_detection.train --config configs/synthetic.yaml

train-ieee:         ## full pipeline on the real Kaggle data (see data/README.md)
	$(PY) -m fraud_detection.train --config configs/ieee.yaml

serve:              ## scoring API on http://localhost:8000 (docs at /docs)
	FRAUD_ARTIFACTS=$(ARTIFACTS) uvicorn fraud_detection.api.app:app --host 0.0.0.0 --port 8000

replay:             ## stream the test month through the API: skew, latency, drift
	$(PY) -m fraud_detection.replay --config $(CONFIG) --limit 5000

test:
	$(PY) -m pytest

lint:
	ruff check src tests && ruff format --check src tests

docker-build:       ## image with the trained bundle baked in (run `make train` first)
	docker build --build-arg ARTIFACTS=$(ARTIFACTS) -t fraud-api .

docker-run:
	docker run --rm -p 8000:8000 fraud-api
