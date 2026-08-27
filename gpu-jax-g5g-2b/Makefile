# gpu-jax-g5g-2b — EC2 G5g (Graviton2 + NVIDIA T4G) pure-JAX rig.
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
	@echo " gpu-jax-g5g-2b — EC2 G5g / NVIDIA T4G pure-JAX rig"
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
	@echo "Serving here is a pip install, not a build: jax[cuda12] publishes"
	@echo "aarch64 wheels whose arch tables carry sm_75. See docs/turing-aarch64-gap.md."

install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m unittest discover -s tests -v

# Lint the rig sources. Snapshots under .claude/skills and skills/ are generated
# copies — lint the sources, not the copies.
#
# ports/ is deliberately excluded. It is a vendored clean-room model port shared
# with tpu-jax-v5e1-2b, and ruff's UP006/UP045 would rewrite its `Dict`/`Optional`
# annotations to `dict`/`X | None` — which the monorepo CLAUDE.md explicitly says
# not to do ("Existing code uses Optional[str], not X | None"). Linting it would
# either fail forever or drift it away from the sibling copy.
#
# NOTE: a new top-level module is silently unlinted until it is added below.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check server.py refresh_skill.py jax_engine.py jax_openai_server.py profile_decode.py profile_prefill.py tune_loop.py tests
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
deploy:
	@echo "Use the MCP tool create_g5g_instance (or get_deployment_config for the CLI form)."
	@echo "Run verify_gpu_arch first — see docs/turing-aarch64-gap.md."
