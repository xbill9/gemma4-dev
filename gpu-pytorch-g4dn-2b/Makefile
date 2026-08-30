# gpu-pytorch-g4dn-2b — EC2 G4dn (x86_64 + NVIDIA T4) PyTorch/transformers rig.
#
# The skill name carries the rig directory: rigs all shipping a skill called
# "tpu-management" meant `make skill-install` rm -rf'd one shared destination, so
# whichever rig ran last silently replaced the rest. Same rule as the plugin and
# MCP server names (NAMING.md).
SKILL_NAME ?= $(notdir $(CURDIR))-management
SKILL_DIR  := .claude/skills/$(SKILL_NAME)
PLUGIN_DIR := skills/$(SKILL_NAME)
DIST_DIR   := dist

.PHONY: all help clean test lint install deploy init skill skill-install skill-package

all: help

help:
	@echo "========================================================="
	@echo " gpu-pytorch-g4dn-2b — EC2 G4dn / NVIDIA T4 PyTorch rig"
	@echo "========================================================="
	@echo "  make install       - pip install -r requirements.txt"
	@echo "  make test          - offline unittest suite"
	@echo "  make lint          - ruff + bash -n on shell scripts"
	@echo "  make clean         - remove caches"
	@echo "  make skill         - refresh $(SKILL_NAME) snapshots"
	@echo "  make skill-install - refresh + copy to ~/.claude/skills"
	@echo "  make skill-package - refresh + build dist/$(SKILL_NAME)-skill.zip"
	@echo "  make init TARGET=/path/to/project [ARGS='--region us-east-1']"
	@echo "========================================================="
	@echo "Torch comes from the AMI, not from pip. Nothing is built on the instance;"
	@echo "the bootstrap installs transformers into the DLAMI's own interpreter."
	@echo "THIS RIG HAS SERVED NOTHING - see docs/INHERITED.md before quoting a number."

install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m unittest discover -s tests -v

# Lint the rig sources. Snapshots under .claude/skills and skills/ are generated
# copies — lint the sources, not the copies.
#
# There is no ports/ exclusion here: this rig vendors no model port. The JAX
# siblings exclude ports/gemma4/ because ruff would rewrite annotations the
# monorepo forbids changing; nothing here is shared that way, so everything at
# the top level is linted.
#
# NOTE: a new top-level module is silently unlinted until it is added below.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check server.py refresh_skill.py torch_generate.py torch_openai_server.py tests
	@for s in project-setup.sh init.sh set_env.sh save-aws-creds.sh; do bash -n $$s || exit 1; done
	@echo "lint OK"

clean:
	rm -rf __pycache__ .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

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

# No `deploy` target on purpose. Provisioning goes through the MCP server's
# create_g4dn_instance, which resolves the AMI from SSM at launch time — a
# Makefile recipe would have to hardcode an id, which is exactly the bug
# inherited from the legacy tips-tree rigs.
deploy:
	@echo "Use the MCP tool create_g4dn_instance (or get_deployment_config for the CLI form)."
	@echo "Run verify_gpu_arch first: torch comes from the AMI, so arch coverage is a"
	@echo "property of the image that booted, not of anything this rig installed."
