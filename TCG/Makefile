SHELL := bash

KAGGLE_USERNAME := $(shell kaggle config view 2>/dev/null | python3 -c "import sys; print(next((l.split()[-1] for l in sys.stdin if l.startswith('- username:')), 'atharva2099'))")
KERNEL_SLUG := hardread-tcg-baseline
BCPPO_SLUG := hardread-tcg-bc-ppo
NOTEBOOK := notebooks/kaggle_runner.ipynb
BCPPO_NOTEBOOK := notebooks/kaggle_bc_ppo.ipynb
DECK := configs/starter_deck.csv
PUSH_DIR := /tmp/tcg-kaggle-push
BCPPO_PUSH_DIR := /tmp/tcg-bcppo-push

.PHONY: setup data smoke eval configs-check local-eval package kaggle-push kaggle-output submit kaggle-push-bcppo kaggle-output-bcppo

## Setup
setup:
	uv venv .venv && uv pip install -r requirements.txt && echo "Ready. Activate: source .venv/bin/activate"

## Data — extract card IDs from raw CSV
data:
	@echo "Processing card data..."
	mkdir -p data/processed && python3 -c "import csv,json; cards=list(csv.DictReader(open('Data/EN_Card_Data.csv'))); ids=[int(r['Card ID']) for r in cards]; json.dump({'card_ids':ids,'total':len(ids)}, open('data/processed/card_ids.json','w')); print(f\"{len(ids)} card IDs saved\")"

## Smoke — run local smoke tests (no cabt needed)
smoke:
	@if [ -d .venv ]; then . .venv/bin/activate && python tests/test_smoke.py; else python tests/test_smoke.py; fi

## Eval — needs Kaggle (cabt not available locally)
eval:
	@echo "cabt is Kaggle-only. Run: make kaggle-push && make kaggle-output"

## configs-check — verify all configs parse and build agents (no cabt needed)
configs-check:
	@if [ -d .venv ]; then . .venv/bin/activate; fi && \
	python3 -c "from src.evaluate import load_config, build_agent_from_config; \
	import glob; [print(f'{p}:', build_agent_from_config(load_config(p)).name) for p in sorted(glob.glob('configs/*.yaml'))]"

## local-eval — run N games locally via Docker (no Kaggle needed)
local-eval:
	@read -p "Games (default 10): " n; n=$${n:-10}; \
	read -p "Agent [bandit/random] (default bandit): " a; a=$${a:-bandit}; \
	docker run --rm --platform linux/amd64 \
	  -v $(PWD):/app -w /app \
	  python:3.11-slim \
	  python scripts/docker_eval.py --n $$n --agent $$a

## Package
package:
	mkdir -p outputs/submissions && cp $(NOTEBOOK) outputs/submissions/submission_v1.ipynb && echo "Packaged: outputs/submissions/submission_v1.ipynb"

## kaggle-push — push notebook to Kaggle and run
kaggle-push: smoke
	rm -rf $(PUSH_DIR) && mkdir -p $(PUSH_DIR) && \
	cp $(NOTEBOOK) $(PUSH_DIR)/ && \
	cp $(DECK) $(PUSH_DIR)/deck.csv && \
	cp notebooks/kernel-metadata.json $(PUSH_DIR)/kernel-metadata.json && \
	python3 -c "import json; md=json.load(open('$(PUSH_DIR)/kernel-metadata.json')); md['id']='$(KAGGLE_USERNAME)/$(KERNEL_SLUG)'; json.dump(md, open('$(PUSH_DIR)/kernel-metadata.json','w'), indent=2); print(f'Kernel ID: {md[\"id\"]}')" && \
	kaggle kernels push -p $(PUSH_DIR) 2>&1

## kaggle-output — pull latest output from Kaggle
kaggle-output:
	mkdir -p outputs/logs && kaggle kernels output $(KAGGLE_USERNAME)/$(KERNEL_SLUG) -p outputs/logs/ 2>&1 && echo "Output pulled to outputs/logs/"

## kaggle-push-bcppo — push the BC+PPO notebook to Kaggle and run
kaggle-push-bcppo:
	rm -rf $(BCPPO_PUSH_DIR) && mkdir -p $(BCPPO_PUSH_DIR) && \
	cp $(BCPPO_NOTEBOOK) $(BCPPO_PUSH_DIR)/ && \
	cp notebooks/kernel-metadata-bc-ppo.json $(BCPPO_PUSH_DIR)/kernel-metadata.json && \
	python3 -c "import json; md=json.load(open('$(BCPPO_PUSH_DIR)/kernel-metadata.json')); md['id']='$(KAGGLE_USERNAME)/$(BCPPO_SLUG)'; json.dump(md, open('$(BCPPO_PUSH_DIR)/kernel-metadata.json','w'), indent=2); print(f'Kernel ID: {md[\"id\"]}')" && \
	kaggle kernels push -p $(BCPPO_PUSH_DIR) 2>&1

## kaggle-output-bcppo — pull latest output from the BC+PPO notebook
kaggle-output-bcppo:
	mkdir -p outputs/logs && kaggle kernels output $(KAGGLE_USERNAME)/$(BCPPO_SLUG) -p outputs/logs/ 2>&1 && echo "Output pulled to outputs/logs/"

## submit — submit the pushed kernel to the competition
submit:
	@echo "Submitting kernel $(KAGGLE_USERNAME)/$(KERNEL_SLUG)..."
	kaggle competitions submit pokemon-tcg-ai-battle -k $(KAGGLE_USERNAME)/$(KERNEL_SLUG) -m "bandit-v1: epsilon-greedy contextual bandit, shaped per-step reward" 2>&1
