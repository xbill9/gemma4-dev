# 🚀 tpu-pytorch-v5e1-12b — TPU Management Skill & MCP Agent

This repository packages the **`tpu-pytorch-v5e1-12b-management` Claude Code skill** and the **`tpu-pytorch-v5e1-12b` MCP server**: an AI DevOps/SRE agent for operating Google Cloud TPU capacity and **Gemma 4** vLLM serving on TPU VMs. It finds and provisions TPU capacity (flex-start VMs, queued resources), starts and debugs vLLM, verifies model health, runs benchmarks, analyzes logs with the self-hosted Gemma 4 model, and tears everything down safely.

**GitHub:** https://github.com/xbill9/tpu-pytorch-v5e1-12b

---

## ⚡ Quick Start — set up a project in one command

`project-setup.sh` installs the `tpu-pytorch-v5e1-12b-management` skill **and** registers the `tpu-pytorch-v5e1-12b` MCP server for any project (idempotent — re-run to refresh):

```bash
./project-setup.sh /path/to/project --project <gcp-project-id>   # one project (.mcp.json + .claude/skills)
./project-setup.sh --global                                      # all projects (~/.claude/skills + user-scope MCP)
make init TARGET=/path/to/project ARGS='--project <id>' # same, refreshing skill snapshots first
```

It uses the system `python3` (warning with a `pip install -r requirements.txt` hint if server dependencies are missing — it never creates a venv), merges the server entry into the project's `.mcp.json` without touching other servers, and prints the remaining manual steps (restart Claude Code, gcloud auth, HF token). See `./project-setup.sh --help` for all options. The installer is also bundled inside the skill itself (`mcp/project-setup.sh`), so an unzipped `dist/tpu-pytorch-v5e1-12b-management-skill.zip` is self-installing.

This repo's **own** `.mcp.json` is gitignored (it embeds your GCP project id) and is generated automatically: `./init.sh` registers the server on first run (leaving an existing entry untouched), or regenerate it any time with `./project-setup.sh . --project <gcp-project-id> [--model ... --accelerator ... --tp ...]`.

---

## 📦 Installing the `tpu-pytorch-v5e1-12b-management` Skill

Claude Code auto-discovers any skill folder containing a `SKILL.md` in two places:

- **Project-level:** `<project>/.claude/skills/tpu-pytorch-v5e1-12b-management/` — available only in that project (this repo ships its own copy, so working inside this repo needs no install).
- **User-level:** `~/.claude/skills/tpu-pytorch-v5e1-12b-management/` — available in every project on the machine.

Pick the install path that fits:

| Goal | Command |
| :--- | :--- |
| This machine, all projects | `make skill-install` |
| One specific project (skill **and** `tpu-pytorch-v5e1-12b` MCP server) | `make init TARGET=/path/to/project ARGS='--project <gcp-project-id>'` |
| All projects + user-scope MCP registration | `make init ARGS='--global'` |
| Another machine | `make skill-package`, copy `dist/tpu-pytorch-v5e1-12b-management-skill.zip`, unzip into `~/.claude/skills/` |

### Install from GitHub

**Option A — Claude Code plugin marketplace (recommended):**

```
/plugin marketplace add xbill9/tpu-pytorch-v5e1-12b
/plugin install tpu-pytorch-v5e1-12b-management@tpu-pytorch-v5e1-12b
```

This installs the `tpu-pytorch-v5e1-12b-management` skill **and** registers the `tpu-pytorch-v5e1-12b` MCP server in one step, with updates managed by Claude Code (`/plugin` → manage/update). Configure the server through environment variables (e.g. `GOOGLE_CLOUD_PROJECT`, `MODEL_NAME`, `ACCELERATOR_TYPE`) — see `SKILL.md` or the `get_help` tool for the full list.

**Option B — clone and install** (all projects on this machine):

```bash
git clone https://github.com/xbill9/tpu-pytorch-v5e1-12b
cd tpu-pytorch-v5e1-12b
make skill-install                                   # skill only
./project-setup.sh --global                          # skill + user-scope tpu-pytorch-v5e1-12b MCP server
```

**Option C — zip install, no clone** (straight from the packaged zip):

```bash
curl -L -o /tmp/tpu-pytorch-v5e1-12b-management-skill.zip \
  https://github.com/xbill9/tpu-pytorch-v5e1-12b/raw/main/dist/tpu-pytorch-v5e1-12b-management-skill.zip
mkdir -p ~/.claude/skills && unzip -o /tmp/tpu-pytorch-v5e1-12b-management-skill.zip -d ~/.claude/skills/
~/.claude/skills/tpu-pytorch-v5e1-12b-management/mcp/project-setup.sh --global   # optional: register the MCP server
```

All of these first run `make skill` (`refresh_skill.py`), which regenerates the bundled snapshots from the repo-root sources: `server.py`, `project-setup.sh`, and `requirements.txt` are copied into the skill's `mcp/` folder, and `references/tpu-guide.md` is rebuilt from `tpu.md` with the embedded screenshots stripped. `SKILL.md` and the `mcp/startup_script*_template.sh` scripts are hand-maintained and never overwritten.

After installing (or updating), **restart Claude Code** or start a new session so it picks up the skill. Verify with `/skills` — `tpu-pytorch-v5e1-12b-management` should be listed.

Because installs are refresh-and-copy (not symlinks), an installed copy goes stale when `server.py`, `tpu.md`, or `SKILL.md` changes — rerun `make skill-install` (or `make init ...`) after editing those files.

---

## 📂 Repository Layout

| Path | Purpose |
| :--- | :--- |
| `server.py` | The `tpu-pytorch-v5e1-12b` FastMCP server — the authoritative source (full tool catalog in `SKILL.md` / the `get_help` tool) |
| `project-setup.sh` | One-command installer: skill + MCP registration for a target project |
| `refresh_skill.py` | Regenerates the bundled skill snapshots from the repo-root sources |
| `requirements.txt` | Python dependencies for the MCP server |
| `Makefile` | `skill` / `skill-install` / `skill-package` / `init` targets (see below) |
| `.claude/skills/tpu-pytorch-v5e1-12b-management/` | Project-level skill: `SKILL.md`, `mcp/` (server snapshot, installer, startup script template), `references/tpu-guide.md` |
| `.claude-plugin/` | `plugin.json` + `marketplace.json` — the repo doubles as a Claude Code plugin marketplace |
| `skills/tpu-pytorch-v5e1-12b-management/` | Plugin-layout copy of the skill (synced by `make skill`) |
| `dist/tpu-pytorch-v5e1-12b-management-skill.zip` | Packaged skill for zip installs (built by `make skill-package`) |
| `init.sh`, `set_env.sh`, `set_adc.sh` | GCP environment / credentials setup helpers |
| `tpu.md` | TPU getting started guide source (gitignored; the stripped, vendor-neutral text copy ships in `references/tpu-guide.md`) |

---

## 🛠 Features & Capabilities

The `tpu-pytorch-v5e1-12b` MCP server covers the full TPU serving lifecycle (catalog with usage guidance in [SKILL.md](.claude/skills/tpu-pytorch-v5e1-12b-management/SKILL.md), live listing via the `get_help` tool):

- **Capacity discovery & provisioning:** sweep zones for available capacity (`find_tpu_vm` for flex-start VMs, `find_tpu` for queued resources), check quotas (`get_zones_with_available_quota`), estimate cost, create flex-start TPU VMs (v6e/v5p) or legacy queued resources (v5e) with an auto-serving startup script, then `wait_for_vllm_ready` until the model is up.
- **Serving stack control:** manage the vLLM Docker container (`manage_vllm_docker` — works on both flex-start VMs and queued-resource nodes), fetch endpoints and the gcloud deployment one-liner, store the HF token in Secret Manager.
- **Health, logs & diagnostics:** system status dashboard covering both serving paths, model health verification, vLLM/docker/system/serial logs, Cloud Logging retrieval, and Gemma-4-powered log triage (`analyze_cloud_logging`).
- **Inference & benchmarking:** query the deployed Gemma 4 endpoint (optional TTFT/throughput stats), run `vllm bench serve` for benchmark metrics.
- **Universal SRE help:** a standardized `get_help` tool describing the active configuration and all exposed tools.

---

## 🏗 Makefile Usage

```bash
make skill         # Refresh skill snapshots from server.py / tpu.md (also syncs the plugin copy in skills/)
make skill-install # Refresh + copy the skill to ~/.claude/skills (all projects)
make skill-package # Refresh + build dist/tpu-pytorch-v5e1-12b-management-skill.zip
make init TARGET=/path/to/project [ARGS='--project my-gcp-id']
                   # Refresh + install skill AND register the tpu-pytorch-v5e1-12b MCP server
```

Edit the repo-root sources (`server.py`, `tpu.md`, `project-setup.sh`), then run the appropriate target — never edit the snapshot copies directly.

---

## ⚙️ Configuration

The server reads its configuration from environment variables: `GOOGLE_CLOUD_PROJECT` (falls back to the active gcloud config), `GOOGLE_CLOUD_ZONE` (default `europe-west4-a`), `GOOGLE_CLOUD_REGION`, `MODEL_NAME`, `ACCELERATOR_TYPE`, and `TENSOR_PARALLEL_SIZE`.

Provisioning a `workload="pytorch"` VM additionally requires the PyTorch TPU backend
install config, which ships with **no defaults** because those packages are not publicly
distributable — set them in the MCP server's environment: `TPU_BACKEND_PIP_SPEC` (package
plus the matching `torch==<ver>+cpu` pin), `TPU_BACKEND_INDEX` (authenticated pip index
host), and optionally `TPU_BACKEND_WHEELS_GCS` (project-owned wheel mirror bucket, tried
before the index), `TPU_BACKEND_PIP_EXTRAS`, and `TPU_BACKEND_ENV_EXPORTS` (space- or
newline-separated `KEY=VALUE` pairs appended to the VM's `/etc/environment` — this is how
the backend's persistent compile-cache variables are supplied). Creation aborts with an
explicit message when the first two are unset. Prerequisites: `pip install -r requirements.txt`, an authenticated `gcloud` CLI with alpha components, the TPU API enabled, and — for vLLM serving only — a Hugging Face token stored as Secret Manager secret `hf-token` (the `save_hf_token` tool does this for you).

---

## 🔒 Security & Credentials

When deploying to Google Cloud or Hugging Face, secure credentials using:
- **Hugging Face Access Token:** Saved locally or to Google Secret Manager.
- **Application Default Credentials (ADC):** Set up using GCP credentials helper scripts (`set_adc.sh`).

## 📖 Related Documentation

- [SKILL.md](.claude/skills/tpu-pytorch-v5e1-12b-management/SKILL.md) — the skill itself: lifecycle, tool catalog, required vLLM flags, field notes, cautions
- [GEMINI.md](GEMINI.md) — Gemini CLI integration via a LiteLLM proxy pointed at the self-hosted Gemma 4 TPU endpoint
- `references/tpu-guide.md` — TPU getting started guide: flex-start zones, quotas, troubleshooting

## Credits

Google Cloud credits are provided for this project.

#AgenticArchitect #GoogleAntigravity
