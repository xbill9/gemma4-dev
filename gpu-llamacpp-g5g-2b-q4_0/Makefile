# gpu-llamacpp-g5g-2b-q4_0 — EC2 G5g (Graviton2 + NVIDIA T4G), llama.cpp + Q4_0 GGUF.
#
# SKILL_NAME derives from the directory, so the rename to this name carried the
# skill, plugin and cache names with it automatically. Do not hardcode it back.
#
# The skill name carries the rig directory: rigs all shipping a skill called
# "tpu-management" meant `make skill-install` rm -rf'd one shared destination, so
# whichever rig ran last silently replaced the rest. Same rule as the plugin and
# MCP server names (NAMING.md).
SKILL_NAME ?= $(notdir $(CURDIR))-management
SKILL_DIR  := .claude/skills/$(SKILL_NAME)
PLUGIN_DIR := skills/$(SKILL_NAME)
DIST_DIR   := dist

.PHONY: all help clean test lint install deploy init skill skill-install skill-package medium

all: help

help:
	@echo "========================================================="
	@echo " gpu-llamacpp-g5g-2b-q4_0 — EC2 G5g / T4G, llama.cpp + Q4_0 GGUF"
	@echo "========================================================="
	@echo "  make install       - pip install -r requirements.txt"
	@echo "  make test          - offline unittest suite"
	@echo "  make lint          - ruff + bash -n on shell scripts"
	@echo "  make clean         - remove caches"
	@echo "  make skill         - refresh $(SKILL_NAME) snapshots"
	@echo "  make skill-install - refresh + copy to ~/.claude/skills"
	@echo "  make skill-package - refresh + build dist/$(SKILL_NAME)-skill.zip"
	@echo "  make medium        - render medium-*.md to paste-ready HTML"
	@echo "  make init TARGET=/path/to/project [ARGS='--region us-east-1']"
	@echo "========================================================="
	@echo "Serving here IS a build: llama.cpp publishes no prebuilt Linux aarch64"
	@echo "CUDA binary, so cloud-init compiles it with CMAKE_CUDA_ARCHITECTURES=75."

# Installs the MCP server's own dependencies, on YOUR machine. Nothing here is
# installed on the instance: the box gets apt build deps and a compiler, and the
# serving artifact is a binary. There is no requirements file for the runtime.
install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m unittest discover -s tests -v

# Lint the rig sources. Snapshots under .claude/skills and skills/ are generated
# copies — lint the sources, not the copies.
#
# There is no ports/ here and no serving payload of our own — the engine is a
# binary built on the box. So this list is just the rig's own tooling.
#
# NOTE: a new top-level module is silently unlinted until it is added below.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check server.py refresh_skill.py sweep.py make_report.py tests
	@for s in project-setup.sh init.sh set_env.sh save-aws-creds.sh; do bash -n $$s || exit 1; done
	@echo "lint OK"

clean:
	rm -rf __pycache__ .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Medium strips markdown tables and reflows monospace, so tables and ASCII
# diagrams are rendered to PNG and inlined as data: URIs. Open the -embed.html
# in a browser, Select All, Copy, paste into the Medium editor.
medium:
	@rm -rf medium
	@for f in medium-*.md; do python3 make-medium.py "$$f" medium; done
	@echo "---------------------------------------------------------"
	@echo "Open in a browser, Ctrl-A, Ctrl-C, paste into Medium:"
	@ls -1 medium/*-embed.html 2>/dev/null | sed 's/^/  /'

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
# create_g5g_instance, which resolves an arm64 AMI at launch time — a Makefile
# recipe would have to hardcode one, and a hardcoded x86_64 AMI id is exactly
# the bug inherited from the legacy tips-tree rigs.
# There is no deploy step at all on this rig, not merely no Makefile target for
# one: cloud-init builds llama.cpp and enables the unit itself, and llama-server
# fetches its own checkpoint. Nothing of ours is shipped to the box.
deploy:
	@echo "Nothing to deploy — cloud-init builds and starts the service."
	@echo "Use create_g5g_instance, then get_install_progress, then verify_gpu_arch."
