# Intro to Cloud TPU

A talk on serving Gemma 4 on Cloud TPU, ending in a live demo: Gemma 4 E2B on one v5e chip, provisioned and queried through the MCP server in [`tpu-vllm-v5e1-2b`](../../tpu-vllm-v5e1-2b).

- Slides: [Google Slides](https://docs.google.com/presentation/d/1bMtCHpNWLK2ys_ZUGFl0V9g7RjNIsFajSHg46DTYTIQ/edit), 29 slides, DevFest 2026 / GDE style
- Copies in this directory, exported 2026-10-24: [`intro-to-cloud-tpu.pdf`](intro-to-cloud-tpu.pdf), [`intro-to-cloud-tpu.pptx`](intro-to-cloud-tpu.pptx)
- Reference article: [The unofficial TPU migration guide: Cloud TPU API to Compute Engine](https://dev.to/gde/the-unofficial-tpu-migration-guide-cloud-tpu-api-to-compute-engine-2co7)

The Google Slides deck is the source. The PDF and PPTX are snapshots; export again after editing it.

## Demo

The demo chip is a `v5litepod-1` flex-start queued resource in `us-west4-a`. It deletes itself 4 hours after it starts, and vLLM takes about 20 minutes to become healthy from a cold VM, so provision it about 30 minutes before the talk.

Run the MCP client from the rig directory. `make mcp-config` writes the `.mcp.json` that registers the server as `tpu-vllm-v5e1-2b`.

| # | Ask the agent | Tool it calls |
|---|---|---|
| 1 | What's running in the zone? | `get_system_status` |
| 2 | Which zones have v5e quota? | `get_zones_with_available_quota` |
| 3 | Create the v5e-1, flex-start | `create_tpu_queued_resource` |
| 4 | Is it up yet? Show the container log | `describe_queued_resource`, `get_vllm_docker_logs` |
| 5 | Is the model healthy? | `verify_model_health` |
| 6 | Explain a TPU in two sentences, with speed | `query_queued_gemma4_with_stats` |
| 7 | Any TPU errors in the last hour? | `analyze_cloud_logging` |
| 8 | Tear it down | `destroy_queued_resource` |

Provision with `create_tpu_queued_resource`. `manage_queued_resource` deletes every other queued resource in the zone, and `us-west4-a` holds the sibling rigs' resources.

`estimate_deployment_cost` returned HTTP 403 from the Billing Catalog API on 2026-10-24, so it is left out of the script.

## Rebuilding the slides

`build/` holds the scripts that built the deck through the Slides API:

- `build_deck.py OUT.json` writes the `batchUpdate` requests. It works on a copy of the DevFest 2026 template deck (`17rd65NCbRdajaF_5NKaUk7WxKc7KeRiTm-9hBfuRS_o`), duplicating its title, section and logo slides by object id, then deleting the template's own slides.
- `push.py PRESENTATION_ID OUT.json` sends the requests as one atomic `batchUpdate`.
- `thumbs.py PRESENTATION_ID OUT_PREFIX [slide_ids]` renders slide thumbnails into contact sheets for review.

The scripts call the API with `gcloud auth print-access-token`, which needs the Drive scope (`gcloud auth login --enable-gdrive-access`), and bill quota to `aisprint-491218`, where `slides.googleapis.com` is enabled.

Copy the template into My Drive before building: its own folder is read-only to this account.
