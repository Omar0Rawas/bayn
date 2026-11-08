# Bayn task runner for macOS / Linux. Windows: use scripts/bayn.ps1 (same targets).
PYTHON ?= python3
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: help setup demo run check bridge ui test doctor clean

help:
	@echo "setup   create .venv, install pinned deps, run doctor"
	@echo "demo    simulated EEG with scripted gestures + dashboard (no hardware)"
	@echo "run     real Muse 2 (default) + dashboard (ARGS='--source lsl')"
	@echo "check   connect to the headset 10 s, report signal quality (ARGS='--name Muse-XXXX')"
	@echo "bridge  EEG bridge only    ui  dashboard only"
	@echo "test    run tests          doctor  check this machine"
	@echo "clean   remove caches"

$(PY):
	$(PYTHON) -m venv $(VENV)

setup: $(PY)
	$(PY) -m pip install --upgrade pip --quiet
	$(PY) -m pip install -r requirements/dev.txt --quiet
	$(PY) -m pip install -e . --no-deps --quiet
	@test -f .env || cp .env.example .env
	$(PY) -m bayn.doctor

demo:
	$(PY) -m bayn.run --demo $(ARGS)

run:
	$(PY) -m bayn.run $(ARGS)

check:
	$(PY) -m bayn.eeg.check $(ARGS)

bridge:
	$(PY) -m bayn.eeg.bridge $(ARGS)

ui:
	$(PY) -m bayn.ui.server

test:
	$(PY) -m pytest $(ARGS)

doctor:
	$(PY) -m bayn.doctor

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name '*.egg-info' \) -prune -exec rm -rf {} +
