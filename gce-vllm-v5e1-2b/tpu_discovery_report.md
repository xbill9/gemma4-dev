# 🚀 GCP TPU Resource Discovery Report
**Project:** `aisprint-491218`

This report spans **both control planes**. The first section is the one this rig provisions into; the two after it belong to the Cloud TPU API and are shown because sibling rigs compete for the same physical chips.

## 🧩 TPU Instances on Compute Engine (this rig's namespace)
| Instance Name | Zone | Machine Type | Status | Rig | Provisioning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **gce-vllm-v6e1-2b** | `europe-west4-a` | `ct6e-standard-1t` | `RUNNING` | `gce-vllm-v6e1-2b` | `FLEX_START` |

## 🖥️ Cloud TPU API VM Nodes (other control plane)
_No TPU VM instances found in the project._

## 🐳 TPU Queued Resources (Flex-start)
| Resource ID | Zone | Accelerator Type | Topology | State |
| :--- | :--- | :--- | :--- | :--- |
| **jax-gemma4-qr** | `` | `None` | `None` | `SUSPENDED` |
| **torchtpu-v5e1-qr** | `` | `None` | `None` | `SUSPENDED` |

## 📊 Available TPU v5e Quotas
Below are the zones where the project has non-zero quota limits for TPU v5e:

| Zone | Limit (Value) |
| :--- | :--- |
| `asia-east1-a` | 512 |
| `asia-east1-b` | 512 |
| `asia-east1-c` | 512 |
| `asia-northeast1-a` | 512 |
| `asia-northeast1-b` | 512 |
| `asia-northeast1-c` | 512 |
| `asia-southeast1-a` | 512 |
| `asia-southeast1-b` | 512 |
| `asia-southeast1-c` | 512 |
| `europe-north1-a` | 512 |
| `europe-north1-b` | 512 |
| `europe-north1-c` | 512 |
| `europe-west1-b` | 512 |
| `europe-west1-c` | 512 |
| `europe-west1-d` | 512 |
| `europe-west3-a` | 512 |
| `europe-west3-b` | 512 |
| `europe-west3-c` | 512 |
| `europe-west4-a` | 512 |
| `europe-west4-b` | 512 |
| `europe-west4-c` | 512 |
| `northamerica-northeast1-a` | 512 |
| `northamerica-northeast1-b` | 512 |
| `northamerica-northeast1-c` | 512 |
| `southamerica-west1-a` | 512 |
| `southamerica-west1-b` | 512 |
| `southamerica-west1-c` | 512 |
| `us-central1-a` | 512 |
| `us-central1-b` | 512 |
| `us-central1-c` | 512 |
| `us-central1-f` | 512 |
| `us-east1-b` | 512 |
| `us-east1-d` | 512 |
| `us-east4-a` | 512 |
| `us-east4-b` | 512 |
| `us-east4-c` | 512 |
| `us-south1-a` | 512 |
| `us-south1-b` | 512 |
| `us-south1-c` | 512 |
| `us-west1-a` | 512 |
| `us-west1-b` | 512 |
| `us-west1-c` | 512 |
| `us-west4-a` | 512 |
| `us-west4-c` | 512 |