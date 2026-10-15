# One Strands agent, every Azure inference path this subscription can run

The Azure third of a three-cloud survey of LLM inference ([aws-inference-strands](https://github.com/xbill9/aws-inference-strands), [gcp-inference](https://github.com/xbill9/gcp-inference), and the comparison in [cross-cloud-inference](https://github.com/xbill9/cross-cloud-inference)). Same demos, same grader, same prompts, same Gemma 4 repack on the self-hosted path.

| Backend | What serves the model | Model |
|---|---|---|
| `foundry` | Azure AI Foundry, global deployment, per token | `gpt-4.1-nano` |
| `phi` | Azure AI Foundry, global deployment, per token | `Phi-4-mini-instruct` |
| `vm` | vLLM 0.30.0 on a `Standard_NV36ads_A10_v5` (one A10, GRID driver), eastus | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` |
| `aca` | vLLM on Container Apps serverless GPU (T4), westus3 | the same repack; **blocked: GPU quota 0** |

| Script | What it does |
|---|---|
| `demo1_counting.py` | The same tool-using agent and question on each backend, graded on the answer **and** on the filter the model sent |
| `demo2_orchestrator.py` | A Foundry agent that calls every backend as a tool and quotes a scoreboard computed in code |
| `bench_chat.py` | The same 100-token prompt on every backend, timed in code |
| `cost.py` | USD per hour and per million output tokens, from Azure Retail Prices API list prices and measured throughput |
| `mcp_server.py` | An MCP server over the same backends: list, health check, logs, ask, launch, tear down, and demo 1's `count_ids` engine |

The three files above are the ones in the AWS and GCP repos; only `backends.py` changes per cloud.

## Bring the backends up

```bash
stage/up.sh vm 2>&1 | tee stage/up.log    # NV36ads_A10_v5 in eastus; starts billing (USD 3.20/hr list)
stage/forward.sh && source stage/demo.env  # SSH tunnel to the VM, Foundry key, writes the variables
python3 stage/check.py --wait 1800         # one question and one tool call per backend; READY
python3 stage/ask.py vm                    # one backend, one question, timed
stage/down.sh                              # delete the VM and the app; --all also deletes the RG and the Foundry deployments
```

The VM boots Ubuntu 24.04, installs NVIDIA's GRID driver (`stage/vm-setup.sh`; the NVadsA10 v5 sizes are vGPU-backed and the datacenter driver does not bind), the container toolkit and vLLM. vLLM listens on the VM's localhost with an API key; the only inbound rule is SSH from the address that ran `up.sh`.

`stage/image/` is vLLM 0.30.0 plus the Turing shared-memory clamp from `~/sagemaker-gemma/turing`, built with `az acr build` for the Container Apps T4 profile.

## MCP server

`mcp_server.py` is the Azure port of the servers in [aws-inference-strands](https://github.com/xbill9/aws-inference-strands) and [gcp-inference](https://github.com/xbill9/gcp-inference): the same tools and annotations, with the `az` CLI in place of boto3, and `az vm run-command` in place of SSM Run Command. `.mcp.json` registers it for Claude Code as `strands-demo`; any MCP client can start it with `python3 mcp_server.py`.

| Tool | Kind | What it does |
|---|---|---|
| `list_backends` | read-only | the VM and Container App tagged `ManagedBy=cci-demo-run` in `cci-demo-rg`, with power state and replica floor, the two Foundry deployments, and how many bill by the hour |
| `backend_health` | read-only | asks one backend for the capital of France and passes only on Paris |
| `backend_logs` | read-only | the vLLM log of `vm` or `aca` |
| `ask_backend` | read-only | one timed chat request, as `stage/ask.py`; uses the variables in `stage/demo.env` |
| `count_ids` | read-only | demo 1's engine: exact count, minimum and maximum for `id <op> value` |
| `launch_log` | read-only | tails `stage/up.log` or `stage/down.log` |
| `launch_backends` | write | runs `stage/up.sh` in the background; billing starts |
| `teardown` | destructive | runs `stage/down.sh` in the background, without `--all` |

Health checks and logs need no tunnel open:

| Backend | `backend_health` | `backend_logs` |
|---|---|---|
| vm | on the VM as root with `az vm run-command`, vLLM's key read from `/etc/cci.env` | `docker logs vllm`, else `/var/log/cci-setup.log` |
| aca | the app's ingress with `stage/.vllm-key` | `az containerapp logs show`, console, else system |
| foundry, phi | the Foundry endpoint with the account key | none |

Launch and teardown run the stage scripts, so both routes start and stop the same resources. With `--tools mcp`, demo 1 offers the agent only `count_ids` from this server and grades the calls it reads back from the agent's messages, so it is graded exactly as `--tools engine`:

```bash
python3 demo1_counting.py --tools mcp
```

## What this subscription allowed (2026-10-12)

| Path | Quota | Result |
|---|---|---|
| VM, NVadsA10 v5 family | 36 vCPUs in eastus, 6 in southcentralus | one NV36ads_A10_v5 |
| VM, NCasT4_v3 and NDasrA100_v4 families | 0 | not run |
| Container Apps serverless GPU, T4 and A100 | 0 for both (`ManagedEnvironmentConsumptionT4Gpus`, `...NCA100Gpus`) | create refused: `ContainerAppInvalidScaleExceedsQuota` |
| Azure ML managed online endpoints | GPU quota only in the retired NC and NV families | not run |
| Foundry per-token | gpt-4.1-nano at 50K TPM; Phi-4-mini capped at 20K TPM | both deployed |

Evidence is in `results/2026-10-12/`.
