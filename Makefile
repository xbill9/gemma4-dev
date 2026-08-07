# Root Makefile for managing all subdirectories

# Dynamically discover all subdirectories containing a Makefile
SUBDIRS := $(patsubst %/,%,$(dir $(wildcard */Makefile)))

# Rigs that carry a benchmarks/ directory. Discovered, not listed, so a new rig is picked up
# by creating the directory rather than by editing this file.
BENCH_RIGS := $(patsubst %/benchmarks/.,%,$(wildcard */benchmarks/.))

.PHONY: all clean test lint install deploy help benchmarks benchmarks-sync benchmarks-rollup benchmarks-validate $(SUBDIRS)

# Default target displays help information
all: help

help:
	@echo "========================================================="
	@echo " Gemma-4 DevOps Agents - Root Makefile"
	@echo "========================================================="
	@echo "Available commands:"
	@echo "  make clean   - Run 'make clean' in all subdirectories"
	@echo "  make test    - Run 'make test' in all subdirectories"
	@echo "  make lint    - Run 'make lint' in all subdirectories"
	@echo "  make install - Run 'make install' in all subdirectories"
	@echo "  make deploy  - Run 'make deploy' in all subdirectories"
	@echo "---------------------------------------------------------"
	@echo "  make benchmarks          - sync + rollup + validate"
	@echo "  make benchmarks-sync     - push the canonical schema/README into every rig"
	@echo "  make benchmarks-rollup   - regenerate ROLLUP.md and each rig's INDEX.md"
	@echo "  make benchmarks-validate - validate every report against the schema"
	@echo "========================================================="

# --- Benchmarks -------------------------------------------------------------------
# benchmarks/serving-report.schema.json and benchmarks/README.md are canonical; each rig
# holds a synced copy. Edit the root files, then `make benchmarks-sync`. A rig's reports/
# and runs/ are its own and are never touched by the sync.
benchmarks: benchmarks-sync benchmarks-rollup benchmarks-validate

benchmarks-sync:
	@set -e; for d in $(BENCH_RIGS); do \
		cp benchmarks/serving-report.schema.json $$d/benchmarks/serving-report.schema.json; \
		cp benchmarks/README.md $$d/benchmarks/README.md; \
		echo "  synced -> $$d/benchmarks/"; \
	done
	@echo "✅ schema + README synced into $(words $(BENCH_RIGS)) rigs"

benchmarks-rollup:
	@python3 benchmarks/rollup.py

# Prefers the check-jsonschema CLI; falls back to the jsonschema module, which is what
# rollup.py already uses. Neither is installed by this target — see benchmarks/README.md.
benchmarks-validate:
	@python3 benchmarks/validate.py

# Target-specific variable assignments
clean: TARGET := clean
clean: $(SUBDIRS)

test: TARGET := test
test: $(SUBDIRS)

lint: TARGET := lint
lint: $(SUBDIRS)

install: TARGET := install
install: $(SUBDIRS)

deploy: TARGET := deploy
deploy: $(SUBDIRS)

# Run the specified target in each subdirectory if a Makefile exists
$(SUBDIRS):
	@if [ -f $@/Makefile ]; then \
		if [ -z "$(TARGET)" ]; then \
			echo "⚙️ Executing default target in $@..."; \
			$(MAKE) -C $@; \
		else \
			echo "⚙️ Executing 'make $(TARGET)' in $@..."; \
			$(MAKE) -C $@ $(TARGET); \
		fi \
	fi
