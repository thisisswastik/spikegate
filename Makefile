# SpikeGate Makefile
# Windows users: use PowerShell or run commands directly

.PHONY: test test-phase0 test-phase1 test-phase2 test-phase3 test-phase4 eval dashboard install

install:
	uv pip install -e ".[dev]"

test:
	pytest tests/ -v

test-phase0:
	pytest tests/test_phase0_imports.py -v

test-phase1:
	pytest tests/test_phase1_datagen.py -v

test-phase2:
	pytest tests/test_phase2_detector.py -v

test-phase3:
	pytest tests/test_phase3_agent.py -v

test-phase4:
	pytest tests/test_phase4_eval.py -v

eval:
	python -m eval.run_eval

dashboard:
	streamlit run dashboard/app.py --server.port 8501

lint:
	python -m py_compile data_gen/*.py detector/*.py agent/**/*.py eval/*.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
