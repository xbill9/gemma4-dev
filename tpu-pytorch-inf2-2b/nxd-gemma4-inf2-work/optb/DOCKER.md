# Prebuilt images — Gemma-4-E2B on inf2 (`xbill9/gemma4-optb`)

Every tag is self-contained: compiled neffs, the real E2B checkpoint, transformers 5.13,
torch_neuronx 2.8.0.2.12 and the Neuron runtime (libnrt) are baked in, version-pinned so the
neffs load. Serving is OpenAI-compatible on port 8080. The images are public on Docker Hub, so
no registry login is needed.

## Tags

Read from each tag's image config on 2026-10-05. Context is `KV_MAX` total tokens and
`KV_BUCKET` prompt tokens. Speeds are single-stream decode, quoted from the commits that built
each image.

| Tag | Server | Context / prompt | NeuronCores | Host | Decode |
| --- | --- | --- | --- | --- | --- |
| `tp2-slim` | `optb_server_tp_slim.py` | 2048 / 512 | 2 (TP=2) | inf2.xlarge + swap | ~61 tok/s |
| `tp2-2048` = `latest` | `optb_server_tp.py` | 2048 / 512 | 2 (TP=2) | inf2.8xlarge | 72.7 tok/s |
| `slim` | `optb_server_slim.py` | 512 / 128 | 1 | inf2.xlarge + swap | `tp2-slim` is 2.5x faster |
| `2048-512` | `optb_server.py` | 2048 / 512 | 1 | inf2.8xlarge | 25 tok/s |
| `512-128` | `optb_server.py` | 512 / 128 | 1 | inf2.8xlarge | ~44 tok/s |

`slim` is what the rig's MCP server deploys by default (`OPTB_IMAGE`). The two `*slim` servers
keep only the embedding and per-layer tables on the host, in bf16, which fits them in
inf2.xlarge's 16 GiB. The neff load peaks near 14.5 GB of host RAM, so that host needs swap.
The other tags load the full fp32 model on the host and need inf2.8xlarge.

## Run

The rig's MCP tool does all of this, swapfile included:
`create_inf2_instance(subnet_id, security_group_id, iam_instance_profile, serving="optb")`,
with `OPTB_IMAGE` set to the tag you want. By hand, on an inf2 host with the Neuron driver
(a Neuron DLAMI):

```bash
# inf2.xlarge only: a 16 GB swapfile for the one-time neff load
sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile

docker run -d --name gemma-optb --restart unless-stopped --ipc=host \
  --device /dev/neuron0 -p 8080:8080 xbill9/gemma4-optb:tp2-slim

curl -s localhost:8080/health
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is AWS Inferentia?"}],"max_tokens":64}'
```

Routes: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health` and `/generate`,
plus `/metrics` and SSE streaming on the TP servers. Sampling takes `temperature`, `top_p` and
`top_k`. There is no auth, so open port 8080 to trusted clients only.

The neff load takes about 90 s from a warm disk. A new EBS volume reads lazily on first touch,
so the first start on a fresh instance can take much longer; the images carry a 20-minute
health-check start period for that reason.

## Rebuilding

The Dockerfiles here build the tags. `Dockerfile` (256/64) and `Dockerfile.512` (512/128) build
single-core images, `Dockerfile.slim` adds the slim server on top of a full image, and
`Dockerfile.tp2` adds the TP server and neffs on top of a single-core 2048/512 image. Its `FROM`
line says `xbill9/gemma4-optb:latest`, which now *is* the TP image; use `:2048-512` as the base
instead. Each Dockerfile `COPY`s artifacts that live outside git: the neffs, the
`real-gemma4-E2B-it` checkpoint, the host's `/opt/aws/neuron` tree and a CPU torch 2.8.0 wheel
(downloaded on the host and copied in, because the build network could not reach PyTorch's
wheel CDN).

The compiled neffs are also on Hugging Face at `xbill9/gemma-4-E2B-it-inferentia2`:
`kv_pre_512.pt`, `kv_dec_512.pt`, `kv_pre_2048.pt`, `kv_dec_2048.pt` and the TP pair under
`tpa_pre/` and `tpa_dec/` (`tp_0.pt` and `tp_1.pt` in each). To compile from scratch, see
`RESTART.md`.

The ECR image (`gemma4-optb:256-64` in us-east-2) that earlier versions of this file named was
deleted by 2026-10-05. The 256/64 build exists in no surviving copy; every Docker Hub tag is
512 or 2048 context.
