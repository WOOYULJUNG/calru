PYTHON ?= python

.PHONY: help install evidence provenance check check-provenance test verify

help:
	@echo "Targets:"
	@echo "  install   Install the aggregation package and test dependency"
	@echo "  evidence  Rebuild committed CSV tables from stored raw evidence"
	@echo "  provenance Rebuild raw file manifest and SHA-256 list"
	@echo "  check     Rebuild in isolation and compare with committed tables"
	@echo "  check-provenance Check that raw provenance files are current"
	@echo "  test      Run the aggregation test suite"
	@echo "  verify    Run reproducibility checks and tests"
	@echo ""
	@echo "Full model retraining is not a verified Make target in this snapshot."

install:
	$(PYTHON) -m pip install -e ".[dev]"

evidence:
	$(PYTHON) scripts/build_evidence.py

provenance:
	$(PYTHON) scripts/build_raw_manifest.py

check:
	$(PYTHON) scripts/check_reproducibility.py

check-provenance:
	$(PYTHON) scripts/build_raw_manifest.py --check

test:
	$(PYTHON) -m pytest -q

verify: check check-provenance test
