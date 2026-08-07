# Root Makefile for managing all subdirectories

# Dynamically discover all subdirectories containing a Makefile
SUBDIRS := $(patsubst %/,%,$(dir $(wildcard */Makefile)))

# The skill name carries the rig directory: five rigs all shipping a skill called
# "tpu-management" meant `make skill-install` rm -rf'd one shared
# ~/.claude/skills/tpu-management, so whichever rig ran last silently replaced the
# rest. Same rule as the plugin and MCP server names (NAMING.md).
SKILL_NAME ?= $(notdir $(CURDIR))-management
SKILL_DIR  := .claude/skills/$(SKILL_NAME)
PLUGIN_DIR := skills/$(SKILL_NAME)
DIST_DIR   := dist

.PHONY: all clean test lint install deploy help init skill skill-install skill-package $(SUBDIRS)

# Default target displays help information
all: help

help:
	@echo "========================================================="
	@echo " AWS Inferentia2 DevOps MCP - Root Makefile"
	@echo "========================================================="
	@echo "Available commands:"
	@echo "  make clean   - Run 'make clean' in all subdirectories"
	@echo "  make test    - Run the repo unit tests (tests/) + 'make test' in subdirectories"
	@echo "  make lint    - ruff + bash -n on repo sources, then 'make lint' in subdirectories"
	@echo "  make install - Run 'make install' in all subdirectories"
	@echo "  make deploy  - Run 'make deploy' in all subdirectories"
	@echo "  make skill         - Refresh $(SKILL_NAME) skill snapshots"
	@echo "                       (also syncs the plugin copy in skills/ for the marketplace)"
	@echo "  make skill-install - Refresh + copy the skill to ~/.claude/skills (all projects)"
	@echo "  make skill-package - Refresh + build dist/$(SKILL_NAME)-skill.zip"
	@echo "  make init TARGET=/path/to/project [ARGS='--region us-east-1']"
	@echo "                     - Refresh + install skill AND register the tpu-pytorch-inf2-2b MCP"
	@echo "                       server in TARGET (or globally with ARGS='--global')"
	@echo "========================================================="

init: skill
	@if [ -z "$(TARGET)" ] && ! echo "$(ARGS)" | grep -q -- --global; then \
		echo "usage: make init TARGET=/path/to/project [ARGS='--region us-east-1 ...']"; \
		echo "   or: make init ARGS='--global ...'"; \
		exit 1; \
	fi
	./project-setup.sh $(TARGET) $(ARGS)

skill:
	SKILL_NAME=$(SKILL_NAME) python3 refresh_skill.py
	rm -rf $(PLUGIN_DIR)
	mkdir -p $(dir $(PLUGIN_DIR))
	cp -r $(SKILL_DIR) $(PLUGIN_DIR)
	@echo "Synced plugin copy -> $(PLUGIN_DIR)"

skill-install: skill
	mkdir -p $(HOME)/.claude/skills
	rm -rf $(HOME)/.claude/skills/$(SKILL_NAME)
	cp -r $(SKILL_DIR) $(HOME)/.claude/skills/$(SKILL_NAME)
	@echo "Installed to $(HOME)/.claude/skills/$(SKILL_NAME)"

skill-package: skill
	mkdir -p $(DIST_DIR)
	rm -f $(DIST_DIR)/$(SKILL_NAME)-skill.zip
	cd $(dir $(SKILL_DIR)) && zip -qr $(CURDIR)/$(DIST_DIR)/$(SKILL_NAME)-skill.zip $(notdir $(SKILL_DIR))
	@echo "Packaged $(DIST_DIR)/$(SKILL_NAME)-skill.zip"
	@unzip -l $(DIST_DIR)/$(SKILL_NAME)-skill.zip

# Target-specific variable assignments
clean: TARGET := clean
clean: $(SUBDIRS)

test: TARGET := test
test: $(SUBDIRS)
	python3 -m unittest discover -s tests -v

# Lint the repo-root sources (snapshots in .claude/skills and skills/ are
# generated copies — lint the sources, not the copies).
lint: TARGET := lint
lint: $(SUBDIRS)
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check server.py refresh_skill.py tests
	@for s in project-setup.sh init.sh set_env.sh set_adc.sh; do bash -n $$s || exit 1; done
	@echo "lint OK"

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
