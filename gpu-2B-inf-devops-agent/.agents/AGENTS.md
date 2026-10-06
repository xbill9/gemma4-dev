# 🤖 Project-Scoped SRE Guidelines: AWS Neuron & vLLM Serving

## ⚡ 1. AWS Neuron Compiler Failure Cache Management
When running vLLM with AWS Neuron (`device=neuron`) on Inferentia (`inf2`) or Trainium (`trn1`) instances, the compiler caches both successful and **failed** compilation graphs (NEFFs). 

* **The Trap**: If a compilation fails once (e.g., due to out-of-memory, disk space exhaustion, or bad input shapes), the Neuron compiler caches that failure in `/var/tmp/neuron-compile-cache/` or inside `.cache/neuron/`. Subsequent launches will immediately fail with `subprocess.CalledProcessError: Command '' died with <Signals.SIGHUP: 1>` without even trying to compile!
* **Container Persistence**: Because the Docker container's root file system persists across standard restarts, any failure cached at `/var/tmp/neuron-compile-cache` inside the container persists and loops infinitely.

### Operational Guardrails & Remedies:
1. **Always Recreate the Container**: Do not rely on `docker restart` or `--restart always` policies alone to recover from compilation crashes. Stop and **remove** the container, then run a fresh container to clear any persistent container-internal `/var/tmp/neuron-compile-cache/` entries.
2. **Purge Cache Directories**: Before redeploying, explicitly purge the following directories on the host:
   ```bash
   sudo rm -rf /var/tmp/neuron-compile-cache
   sudo rm -rf /home/ubuntu/.cache/neuron/*
   ```
3. **Check Host Storage First**: Host storage exhaustion causes compilation failure and subsequent silent SSM and container failures. Reclaim space immediately via:
   ```bash
   docker volume prune -f
   docker system prune -af
   ```

---

## 🧩 2. Gemma 4 Hybrid Attention & KV Cache Head Dimension Matching
Gemma 4's hybrid attention architecture alternates standard sliding-window layers (`head_dim = 256`) and global attention layers (`head_dim = 512`). 

* **The Mismatch**: Because the model's KV Cache is allocated statically to the maximum `head_dim = 512` layer size for all layers, any updates or writes (e.g., during sliding window layers where `head_dim = 256`) can cause an XLA/HLO compilation crash due to shape and memory space mismatches (e.g., `bf16[4,4,4096,256]` versus `bf16[4,4,4096,512]`).
* **The Remedy (Dynamic Update Padding)**: When patching the `neuronx-distributed-inference` modules (specifically `utils.py`), we must apply explicit padding to any incoming updates before they are committed via slice operations:
  
  1. **`update_cache_const_indices`**:
     ```python
     if updates.shape[-1] < d_head:
         updates = torch.nn.functional.pad(updates, (0, d_head - updates.shape[-1]))
     ```
  
  2. **`dynamic_update_slice`**:
     ```python
     if update.shape[-1] < tensor.shape[-1]:
         update = torch.nn.functional.pad(update, (0, tensor.shape[-1] - update.shape[-1]))
     ```

* **The Remedy (Slicing on Retrieval)**: Conversely, when fetching the cached values in the managers (`kv_cache_manager.py`, `gpt_oss_kv_cache_manager.py`, etc.), we must slice the returned cache back from `512` to `256` for sliding-window layers (`(idx + 1) % 6 != 0`):
  ```python
  if (idx + 1) % 6 != 0:
      if k_cache.shape[-1] == 512:
          k_cache = k_cache[..., :256]
      if v_cache.shape[-1] == 512:
          v_cache = v_cache[..., :256]
  ```

* **The Remedy (SWA KV-Cache Position ID Math Constraints - Error 1006)**: When configuring smaller serving context lengths (e.g., `--max-model-len 1024`), the physical KV-cache buffers are dynamically restricted to `1024`. Unpatched sliding window attention modulo calculations (`position_ids % self.sliding_window`) allow position IDs up to `4096`, causing index assignments that exceed `1024` and triggering hardware-level memory faults (Error status 1006). Restrict the modulo base to the minimum of `self.sliding_window` and the actual context sequence allocation dimensions (`seq_dim_size`):
  ```python
  position_ids = position_ids % min(self.sliding_window, seq_dim_size)
  ```

* **The Remedy (Warmup Dimension Alignment - 256 vs 512)**: During token generation warmup, multiplying padded query vectors $Q$ (padded to 512 in preprocessing) with memory-sliced key cache $K_{prior}$ (256) causes a PyTorch/XLA shape crash (`256 vs 512`). To resolve this, dynamically slice $Q, K, V$ back to `256` inside `compute_for_token_gen` when executing local SWA layers:
  ```python
  if getattr(self, "head_dim", 0) == 256:
      if Q.shape[-1] == 512: Q = Q[..., :256]
      if K.shape[-1] == 512: K = K[..., :256]
      if V.shape[-1] == 512: V = V[..., :256]
  ```

* **The Remedy (Sliding Window Silent Disabling Bug)**: Because standard PagedAttention/Triton fallbacks can silently fail to pass sliding window parameters to the decode kernel, the local 512-token SWA can be disabled, leading to severe output hallucinations and looping. To fully mitigate this:
  1. **Strictly Bound Indices:** Ensure index updates in `_get_index_to_update_new_position` are strictly modulated by the SWA limits rather than raw sliding window size.
  2. **Compile-Time Definition:** Ensure compilation scripts define `sliding_window` size to 512 at the static graph compilation stage (rather than 256), as global context operates natively at 512.
  3. **Disable Triton Attention:** Explicitly bypass/disable Triton-attention execution paths, routing instead through custom-patched static Neuron execution paths.

---

## 💾 3. Custom Native Quantization & Context Bounds

When serving larger model parameter counts (like Gemma 4 12B) on tighter physical on-chip memory limits (such as `inf2.xlarge` with exactly one `/dev/neuron0` device containing 2 cores and 32GB total HBM), standard unquantized BF16 execution will overflow memory during staging.

To safely compress memory allocations and serve stably, we outline the native configuration and the fallback legacy workaround.

### A. The Native Method (Recommended)
By default, the AWS Neuron SDK's Distributed Inference (NxD) library handles quantization natively. To avoid "fishy" registry hacks:
1. **Do NOT** set vLLM's `--quantization` CLI setting to `neuron_quant`. Keep the CLI parameter completely unset.
2. Configure native quantization parameters directly via `--additional-config` or using the Neuron compiler beforehand:
   ```bash
   # Pre-compilation with AMP (Automatic Mixed Precision) or native quantization
   python -m transformer_neuronx.export \
     --model google/gemma-4-12b-it \
     --batch_size 1 \
     --amp bf16 \
     --output_dir ./gemma4_compiled_graph
   ```
3. Load the pre-compiled graph cache using `NEURON_COMPILE_CACHE_URL` when booting the container.

---

### B. Fallback Workaround (Programmatic "neuron_quant" Injection)
If your container version/environment rigidly enforces CLI argument validation or lacks external pre-compilation support, the fallback is to inject the dummy config class to satisfy the vLLM validator:

Inject the custom `NeuronQuantConfig` class directly into the container's internal site-packages `/opt/conda/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/__init__.py`:

```python
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization import register_quantization_config
import torch

@register_quantization_config("neuron_quant")
class NeuronQuantConfig(QuantizationConfig):
    def get_name(self) -> str:
        return "neuron_quant"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "NeuronQuantConfig":
        return cls()

    def get_quant_method(self, layer, prefix):
        return None
```

### C. Scaled Context and Memory Parameters
When initializing the serving command, restrict maximum parameters to stay strictly within safety boundaries:
- `--max-model-len 1024` (Prevents excessive KV-cache allocation footprint)
- `--max-num-seqs 2` (Limits concurrent requests dynamically tracking large states)
- `--num-gpu-blocks-override 128` (Sets maximum on-device caches)
- `--max-num-batched-tokens 512` (Lowers prefill processing spike sizes)
- `--tensor-parallel-size 2` (Must map to exactly $2 \times \text{number of visible neuron devices}$ on single cards)

---

## 🛑 4. AWS Neuron Scratchpad Page Size & Allocation Failure Remediation

During the compilation or loading phase of larger heterogeneous models (such as the Gemma 4 families), the AWS Neuron Runtime can run out of chip SRAM/HBM scratchpad memory space, resulting in:
`RuntimeError: Could not load the model status=4 message=Allocation Failure` or `ERROR TDRV:ib_create_eib_impl Failed to create instruction blocks`.

### ⚙️ Mandatory Configuration Overrides:
To prevent scratchpad overflow on instance boot, always increase the scratchpad page size using both compiler flags and execution environment flags:

1. **Compiler Flag (`NEURON_CC_FLAGS`):**
   ```bash
   --hbm-scratchpad-page-size=1024
   ```
   *Example:*
   `export NEURON_CC_FLAGS="--model-type=gemma4 --enable-mixed-shapes=False --target=inf2 --hbm-scratchpad-page-size=1024"`

2. **Execution Environment Flag:**
   ```bash
   export NEURON_SCRATCHPAD_PAGE_SIZE=1024
   ```

Deploying or launching containers without these overrides on tight resource constraints (e.g., `inf2.xlarge` or single-core/dual-core allocations) will trigger silent warm-up loops or immediate runtime restarts.

---

## 💸 5. Strict AWS Cost Containment Guardrails

To prevent excessive cloud spending and comply with budget boundaries, we must maintain a strict single-active-host boundary per project per region.

* **Single Host Constraint**: **Never allow more than one AWS EC2 instance to run concurrently per project per region** for serving or DevOps agents.
* **Pre-Provision Verification**: Before starting or launching any new EC2 resources (via scripts or `deploy_vllm` tools), always invoke `status_ec2` or describe active instances.
* **Immediate Decommission**: If a redundant instance is discovered in a `running` state (e.g. during a hot-swap or automated recovery loop), immediately call `terminate_instances` on the unused instance.

---

## 💾 6. Docker Storage Maintenance & Volume Pruning

When vLLM containers are repeatedly started or crashed during JIT-compilation, they generate anonymous volumes containing model weights and compilation graphs. Standard image and container prunes will NOT delete these, leading to silent 100% disk usage locks.

* **Reclaim Anonymous Volumes**: Always run a volume prune to clean up unused disk space:
  ```bash
  docker volume prune -f
  ```
* **Verify Free Space**: Ensure the root filesystem `/dev/root` has at least 30 GB of free space before starting new vLLM compilation cycles. Check disk space via `df -h`.

