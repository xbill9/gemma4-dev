# gpu-vllm-mi300x-2b — the make targets are the interface.

PYTHON ?= python3

# The skill name carries the rig directory: rigs all shipping a skill called
# "tpu-management" meant `make skill-install` rm -rf'd one shared directory, so
# whichever rig ran last silently replaced the rest. Same rule as the plugin
# and MCP server names (NAMING.md).
SKILL_NAME ?= $(notdir $(CURDIR))-management
SKILL_DIR  := .claude/skills/$(SKILL_NAME)
PLUGIN_DIR := skills/$(SKILL_NAME)
DIST_DIR   := dist

.PHONY: help install lint test check skill skill-install skill-package init clean deploy

help:  ## List targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

install:  ## Install dependencies into the system python3 (no virtualenv)
	$(PYTHON) -m pip install -r requirements.txt

lint:  ## Lint and format-check the rig sources (snapshots are generated copies)
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff format --check server.py refresh_skill.py tests
	ruff check server.py refresh_skill.py tests
	@for s in project-setup.sh init.sh; do bash -n $$s || exit 1; done
	@echo "lint OK"

test:  ## Run the offline unit tests — unittest, never pytest
	$(PYTHON) -m unittest discover -s tests -v

check: lint test  ## What CI would run

skill:  ## Refresh the skill snapshots from the rig-root sources
	SKILL_NAME=$(SKILL_NAME) $(PYTHON) refresh_skill.py
	rm -rf $(PLUGIN_DIR)
	mkdir -p $(dir $(PLUGIN_DIR))
	cp -r $(SKILL_DIR) $(PLUGIN_DIR)
	@echo "Synced plugin copy -> $(PLUGIN_DIR)"

skill-install: skill  ## Refresh + copy the skill to ~/.claude/skills
	mkdir -p $(HOME)/.claude/skills
	rm -rf $(HOME)/.claude/skills/$(SKILL_NAME)
	cp -r $(SKILL_DIR) $(HOME)/.claude/skills/$(SKILL_NAME)
	@echo "Installed to $(HOME)/.claude/skills/$(SKILL_NAME)"

skill-package: skill  ## Refresh + build dist/$(SKILL_NAME)-skill.zip
	mkdir -p $(DIST_DIR)
	rm -f $(DIST_DIR)/$(SKILL_NAME)-skill.zip
	cd $(dir $(SKILL_DIR)) && zip -qr $(CURDIR)/$(DIST_DIR)/$(SKILL_NAME)-skill.zip $(notdir $(SKILL_DIR))
	@echo "Packaged $(DIST_DIR)/$(SKILL_NAME)-skill.zip"

init: skill  ## Install skill + register the MCP server in TARGET
	@if [ -z "$(TARGET)" ] && ! echo "$(ARGS)" | grep -q -- --global; then \
	    echo "usage: make init TARGET=/path/to/project [ARGS='--tag gemma ...']"; \
	    echo "   or: make init ARGS='--global ...'"; \
	    exit 1; \
	fi
	./project-setup.sh $(TARGET) $(ARGS)

deploy:  ## No-op: this rig deploys by MCP tool, not by make
	@echo "Use the MCP server: deploy_vllm. See README.md."

clean:  ## Remove caches
	rm -rf __pycache__ .ruff_cache tests/__pycache__
