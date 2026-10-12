# Common tasks.  `make help` lists them.
.PHONY: help install install-clip data ingest api ui test lint eval image-eval llm-eval up up-monitoring down clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' Makefile | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

install:         ## install API + UI + dev deps (lite backend, no torch)
	pip install -r requirements/dev.txt

install-clip:    ## add the CLIP / sentence-transformers backend (CPU torch)
	pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements/clip.txt

data:            ## regenerate the synthetic catalog + images
	python scripts/generate_sample_data.py

ingest:          ## ETL + index into QDRANT_URL
	python scripts/ingest.py --recreate

api:             ## run the API with hot reload on :8000
	uvicorn app.api.main:app --reload --port 8000

ui:              ## run the Streamlit UI on :8501
	API_URL=$${API_URL:-http://localhost:8000} streamlit run ui/streamlit_app.py

test:            ## unit + integration tests + quality gate
	pytest -q

lint:            ## ruff
	ruff check app ui scripts tests

eval:            ## offline retrieval evaluation (keyword vs semantic vs hybrid vs agent)
	python -m app.evaluation.retrieval_eval

image-eval:      ## image search eval: one photo per boot vs every photo (MaxSim)
	python -m app.evaluation.image_eval

llm-eval:        ## generation eval + prompt A/B (needs ANTHROPIC_API_KEY for the judge)
	python -m app.evaluation.llm_eval

up:              ## full stack in Docker (qdrant, redis, ingest, api, ui)
	docker compose up --build

up-monitoring:   ## full stack + Prometheus + Grafana
	docker compose --profile monitoring up --build

down:            ## stop the Docker stack
	docker compose --profile monitoring down

clean:           ## remove generated state
	rm -rf data/processed data/app.db data/eval/report.json data/eval/llm_report.json data/eval/image_report.json qdrant_data .pytest_cache
