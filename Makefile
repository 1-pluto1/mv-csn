.PHONY: help setup verify test summary demo error-analysis full

PYTHON ?= python3

help: ## Show available one-click reproduction commands
	@echo "MV-CSN v2 One-Click Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-16s %s\n", $$1, $$2}'

setup: ## Install dependencies and editable mvcsn package
	@PYTHON_BIN=$(PYTHON) ./scripts/reproduce.sh setup

verify: ## Fast CPU verification (~10s): compile + tests + live inference + summary
	@PYTHON_BIN=$(PYTHON) ./scripts/reproduce.sh verify

test: ## Run automated pytest suite (tests/)
	@$(PYTHON) -m pytest -v tests/

summary: ## Instantly print all verified experiment tables from results/*.json
	@PYTHON_BIN=$(PYTHON) ./scripts/reproduce.sh summary

demo: ## Launch the 3-tab interactive Streamlit Web Inspector (CPU)
	@PYTHON_BIN=$(PYTHON) ./scripts/reproduce.sh demo

error-analysis: ## Reproduce Exp 7 CPU error analysis (3,395 test SMS)
	@PYTHON_BIN=$(PYTHON) ./scripts/reproduce.sh error-analysis

full: ## Full end-to-end GPU reproduction of Exp 1-7 (~1.5h on A100/L4)
	@PYTHON_BIN=$(PYTHON) ./scripts/reproduce.sh full
