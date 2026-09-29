PYTHON ?= python3
VENV ?= .venv

.PHONY: venv test lint build clean fetch-yolov8n

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -q -U pip
	$(VENV)/bin/pip install -q -r requirements-dev.txt

venv: $(VENV)/bin/activate

test: venv
	$(VENV)/bin/python -m pytest -q

lint: venv
	$(VENV)/bin/pip install -q ruff
	$(VENV)/bin/ruff check src tests

# Same steps the Viam cloud build runs (setup.sh creates ./venv, build.sh produces dist/archive.tar.gz).
build:
	./setup.sh
	./build.sh

# Optional real-model test asset for the opt-in YOLOv8n integration test (see tests/test_yolov8n_real.py).
fetch-yolov8n:
	mkdir -p tests/fixtures
	curl -L -o tests/fixtures/yolov8n.onnx "$${YOLOV8N_ONNX_URL:?set YOLOV8N_ONNX_URL to a yolov8n.onnx download URL}"

clean:
	rm -rf build dist venv .installed *.spec

# Compare CPU / GPU / NPU / AUTO on this machine, e.g. make bench MODEL=/path/to/yolov8n.onnx
bench: venv
	$(VENV)/bin/python scripts/benchmark_devices.py --model "$${MODEL:?set MODEL=/path/to/model}" --markdown --json bench-results.json
