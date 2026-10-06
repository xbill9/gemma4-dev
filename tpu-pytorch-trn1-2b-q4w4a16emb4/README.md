# tpu-pytorch-trn1-2b-q4w4a16emb4

Gemma 4 E2B on one AWS Trainium chip (`trn1.2xlarge`), for the checkpoint `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`: the QAT W4A16 repack with `embed_tokens`, the per-layer embeddings and `lm_head` int4 as well, text only.

This is `tpu-pytorch-trn1-2b` with a different checkpoint, and `tpu-pytorch-inf2-2b-q4w4a16emb4` on Trainium instead of Inferentia2. Both chips have two NeuronCore-v2 cores and 32 GiB of device memory.

## Quick start

```bash
pip install -r requirements.txt
./project-setup.sh . --region us-east-2
```

Then, from Claude Code: `check_trn1_quotas`, `get_deployment_config`, `create_trn1_instance` (pass `spot=False` until a Trn spot quota exists), `verify_neuron_health`.

`create_trn1_instance` serves the prebuilt dense E2B image. To run this rig's checkpoint, install `requirements-serving.txt` on the instance and run the native engine by hand:

```bash
python3 torch_generate.py --model xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4 --parity
python3 bench_arm.py --model xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4 --neff-dir /opt/neff --out arm.json --batches 1,4,16 --steps 96
```

## Measured

Nothing yet.
