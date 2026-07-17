PYTHON ?= python

.PHONY: help status install evidence provenance artifacts artifacts-list \
	check-artifacts check check-provenance check-layout test test-experiment verify

help:
	@echo "Targets:"
	@echo "  status    Show canonical current/auxiliary/archived artifact groups"
	@echo "  install   Install the aggregation package and test dependency"
	@echo "  evidence  Rebuild committed CSV tables from stored raw evidence"
	@echo "  provenance Rebuild raw file manifest and SHA-256 list"
	@echo "  artifacts Sync current paper figures/tables from frozen experiments"
	@echo "  artifacts-list List canonical paper artifact groups"
	@echo "  check-artifacts Verify paper artifacts against local frozen sources"
	@echo "  check     Rebuild in isolation and compare with committed tables"
	@echo "  check-provenance Check that raw provenance files are current"
	@echo "  check-layout Check artifact manifest and canonical documentation"
	@echo "  test      Run lightweight aggregation/repository tests"
	@echo "  test-experiment Run the full suite (requires the GPU-analysis environment)"
	@echo "  verify    Run checks that do not require local frozen experiments"
	@echo ""
	@echo "Full model retraining is not a verified Make target in this snapshot."

status:
	$(PYTHON) scripts/repository_status.py

install:
	$(PYTHON) -m pip install -e ".[dev]"

evidence:
	$(PYTHON) scripts/build_evidence.py

provenance:
	$(PYTHON) scripts/build_raw_manifest.py

artifacts:
	$(PYTHON) scripts/sync_paper_artifacts.py

artifacts-list:
	$(PYTHON) scripts/sync_paper_artifacts.py --list

check-artifacts:
	$(PYTHON) scripts/sync_paper_artifacts.py --check

check:
	$(PYTHON) scripts/check_reproducibility.py

check-provenance:
	$(PYTHON) scripts/build_raw_manifest.py --check

check-layout:
	$(PYTHON) scripts/repository_status.py --check

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q \
		tests/test_evidence.py tests/test_sync_paper_artifacts.py

test-experiment:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q

verify: check check-provenance check-layout test
