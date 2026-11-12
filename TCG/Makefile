SHELL := bash

KAGGLE_USERNAME := $(shell kaggle config view 2>/dev/null | python3 -c "import sys; print(next((l.split()[-1] for l in sys.stdin if l.startswith('- username:')), 'atharva2099'))")
KERNEL_SLUG := hardread-tcg-baseline
BCPPO_SLUG := hardread-tcg-bc-ppo
NOTEBOOK := notebooks/kaggle_runner.ipynb
BCPPO_NOTEBOOK := notebooks/kaggle_bc_ppo.ipynb
PPOBC_SLUG := hardread-tcg-ppo-from-bc
PPOBC_NOTEBOOK := notebooks/kaggle_ppo_from_bc.ipynb
DECK := configs/starter_deck.csv
PUSH_DIR := /tmp/tcg-kaggle-push
BCPPO_PUSH_DIR := /tmp/tcg-bcppo-push
PPOBC_PUSH_DIR := /tmp/tcg-ppo-from-bc-push

.PHONY: setup data smoke eval configs-check package kaggle-push kaggle-output submit kaggle-push-bcppo kaggle-push-bcppo-smoke kaggle-status-bcppo kaggle-output-bcppo kaggle-push-ppo-from-bc kaggle-push-ppo-from-bc-smoke kaggle-status-ppo-from-bc kaggle-output-ppo-from-bc

## Setup
setup:
	uv venv .venv && uv pip install -r requirements.txt && echo "Ready. Activate: source .venv/bin/activate"

## Data — extract card IDs from raw CSV
data:
	@echo "Processing card data..."
	mkdir -p data/processed && python3 -c "import csv,json; cards=list(csv.DictReader(open('Data/EN_Card_Data.csv'))); ids=[int(r['Card ID']) for r in cards]; json.dump({'card_ids':ids,'total':len(ids)}, open('data/processed/card_ids.json','w')); print(f\"{len(ids)} card IDs saved\")"

## Smoke — run local smoke tests (no cabt needed)
smoke:
	uv run python tests/test_smoke.py

## Eval — needs Kaggle (cabt not available locally)
eval:
	@echo "cabt is Kaggle-only. Run: make kaggle-push && make kaggle-output"

## configs-check — verify all configs parse and build agents (no cabt needed)
configs-check:
	@if [ -d .venv ]; then . .venv/bin/activate; fi && \
	python3 -c "from src.evaluate import load_config, build_agent_from_config; \
	import glob; [print(f'{p}:', build_agent_from_config(load_config(p)).name) for p in sorted(glob.glob('configs/*.yaml'))]"

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
	@python3 -c "import json; nb=json.load(open('$(BCPPO_NOTEBOOK)')); src=''.join(''.join(c.get('source', [])) for c in nb['cells']); assert 'SMOKE_MODE = False' in src, 'Refusing full run: set SMOKE_MODE = False first'"
	rm -rf $(BCPPO_PUSH_DIR) && mkdir -p $(BCPPO_PUSH_DIR) && \
	cp $(BCPPO_NOTEBOOK) $(BCPPO_PUSH_DIR)/ && \
	cp notebooks/kernel-metadata-bc-ppo.json $(BCPPO_PUSH_DIR)/kernel-metadata.json && \
	python3 -c "import json; md=json.load(open('$(BCPPO_PUSH_DIR)/kernel-metadata.json')); md['id']='$(KAGGLE_USERNAME)/$(BCPPO_SLUG)'; json.dump(md, open('$(BCPPO_PUSH_DIR)/kernel-metadata.json','w'), indent=2); print(f'Kernel ID: {md[\"id\"]}')" && \
	kaggle kernels push -p $(BCPPO_PUSH_DIR) 2>&1

## kaggle-push-bcppo-smoke — push the BC notebook with SMOKE_MODE=True
kaggle-push-bcppo-smoke:
	@python3 -c "import json; nb=json.load(open('$(BCPPO_NOTEBOOK)')); src=''.join(''.join(c.get('source', [])) for c in nb['cells']); assert 'SMOKE_MODE = True' in src, 'Refusing smoke run: set SMOKE_MODE = True first'"
	rm -rf $(BCPPO_PUSH_DIR) && mkdir -p $(BCPPO_PUSH_DIR) && \
	cp $(BCPPO_NOTEBOOK) $(BCPPO_PUSH_DIR)/ && \
	cp notebooks/kernel-metadata-bc-ppo.json $(BCPPO_PUSH_DIR)/kernel-metadata.json && \
	python3 -c "import json; md=json.load(open('$(BCPPO_PUSH_DIR)/kernel-metadata.json')); md['id']='$(KAGGLE_USERNAME)/$(BCPPO_SLUG)'; json.dump(md, open('$(BCPPO_PUSH_DIR)/kernel-metadata.json','w'), indent=2); print(f'Kernel ID: {md[\"id\"]}')" && \
	kaggle kernels push -p $(BCPPO_PUSH_DIR) 2>&1

## kaggle-status-bcppo — one-shot status check; do not loop/poll
kaggle-status-bcppo:
	kaggle kernels status $(KAGGLE_USERNAME)/$(BCPPO_SLUG)
	@echo "Notebook: https://www.kaggle.com/code/$(KAGGLE_USERNAME)/$(BCPPO_SLUG)"

## kaggle-output-bcppo — pull latest output from the BC+PPO notebook
kaggle-output-bcppo:
	mkdir -p outputs/logs && kaggle kernels output $(KAGGLE_USERNAME)/$(BCPPO_SLUG) -p outputs/logs/ 2>&1 && echo "Output pulled to outputs/logs/"

kaggle-push-ppo-from-bc-smoke:
	@python3 -c "import json; nb=json.load(open('$(PPOBC_NOTEBOOK)')); src=''.join(''.join(c.get('source', [])) for c in nb['cells']); assert 'SMOKE_MODE = True' in src; assert 'n_parallel=4' in src; assert 'ROUNDS = 2 if SMOKE_MODE else 10' in src"
	rm -rf $(PPOBC_PUSH_DIR) && mkdir -p $(PPOBC_PUSH_DIR) && \
	cp $(PPOBC_NOTEBOOK) $(PPOBC_PUSH_DIR)/ && \
	cp notebooks/kernel-metadata-ppo-from-bc.json $(PPOBC_PUSH_DIR)/kernel-metadata.json && \
	python3 -c "import json; p='$(PPOBC_PUSH_DIR)/kernel-metadata.json'; md=json.load(open(p)); md['id']='$(KAGGLE_USERNAME)/$(PPOBC_SLUG)'; json.dump(md, open(p,'w'), indent=2)" && \
	kaggle kernels push -p $(PPOBC_PUSH_DIR) 2>&1

kaggle-push-ppo-from-bc:
	@python3 -c "import json; nb=json.load(open('$(PPOBC_NOTEBOOK)')); src=''.join(''.join(c.get('source', [])) for c in nb['cells']); assert 'SMOKE_MODE = False' in src; assert 'n_parallel=4' in src; assert 'ROUNDS = 2 if SMOKE_MODE else 10' in src"
	rm -rf $(PPOBC_PUSH_DIR) && mkdir -p $(PPOBC_PUSH_DIR) && \
	cp $(PPOBC_NOTEBOOK) $(PPOBC_PUSH_DIR)/ && \
	cp notebooks/kernel-metadata-ppo-from-bc.json $(PPOBC_PUSH_DIR)/kernel-metadata.json && \
	python3 -c "import json; p='$(PPOBC_PUSH_DIR)/kernel-metadata.json'; md=json.load(open(p)); md['id']='$(KAGGLE_USERNAME)/$(PPOBC_SLUG)'; json.dump(md, open(p,'w'), indent=2)" && \
	kaggle kernels push -p $(PPOBC_PUSH_DIR) 2>&1

kaggle-status-ppo-from-bc:
	kaggle kernels status $(KAGGLE_USERNAME)/$(PPOBC_SLUG)

kaggle-output-ppo-from-bc:
	mkdir -p outputs/ppo-v2-smoke && kaggle kernels output $(KAGGLE_USERNAME)/$(PPOBC_SLUG) -p outputs/ppo-v2-smoke/ 2>&1

## submit — submit the pushed kernel to the competition
submit:
	@echo "Submitting kernel $(KAGGLE_USERNAME)/$(KERNEL_SLUG)..."
	kaggle competitions submit pokemon-tcg-ai-battle -k $(KAGGLE_USERNAME)/$(KERNEL_SLUG) -m "bandit-v1: epsilon-greedy contextual bandit, shaped per-step reward" 2>&1
