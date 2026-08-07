# TPU getting started guide

> **Note:** This is a text-only copy of the repository's `tpu.md` (460 KB original). The
> embedded base64 screenshots for the Cloud Console walkthrough have been stripped;
> `![][imageN]` markers show where they appeared. See the original `tpu.md` at the repo
> root for the images.

# TPU Getting Started Guide Last Updated Jul 14, 2026

Welcome\! This guide will walk you through the process of setting up your cloud environment, acquiring TPU capacity, and managing your credits.

## New to TPUs?

* See [How To Scale Your Model](https://jax-ml.github.io/scaling-book/) for a technical overview.

## Prerequisites

Before requesting a TPU, ensure your gcloud CLI is authenticated and the alpha components are installed.

```shell
# 1. Authenticate with your cloud account
gcloud auth login

# 2. Set your Cloud Project ID (replace with your actual project ID)
gcloud config set project YOUR_PROJECT_ID

# 3. Enable the TPU API
# See https://docs.cloud.google.com/endpoints/docs/openapi/enable-api
gcloud services enable tpu.googleapis.com

# 4. Install alpha components for queued resources
gcloud components install alpha --quiet
```

## Run Pre-Flight Diagnostics Script (Recommended) {#run-pre-flight-diagnostics-script-(recommended)}

To proactively verify your billing configuration, default VPC setup, organization policies, API access, service account permissions, and regional/global quotas before attempting a provision, run our automated diagnostic script:

```shell
# Download and run the pre-flight diagnostic tool
curl -sSO https://gist.githubusercontent.com/RobMulla/ee1a530f9ff0bdb9aa5b493c7faf9aa2/raw/tpu_diagnostic.py && python3 tpu_diagnostic.py
```

Running this check can be helpful when diagnosing problems.

## Requesting a TPU

You can request TPU capacity directly as a Compute Engine VM instance using the **`FLEX_START`** provisioning model.

*Note: you can now create a TPU through the cloud console.* [Click here for a walkthrough with screenshots]().

Flex-start capacity is shared among all users in specific zones. You can create a flex-start instance only through the command line, as flex-start is a new feature currently unsupported in the Cloud Console UI. Learn more about [TPU resources in Compute Engine](https://docs.cloud.google.com/compute/docs/tpus/tpu-resources-in-compute-engine) and review the guide on [About Flex-start VMs](https://docs.cloud.google.com/compute/docs/instances/about-flex-start-vms). You can find pricing for these options [here](https://cloud.google.com/products/dws/pricing?e=48754805#flex-start-tpu-vm-pricing).

*(Note: TPU v5e utilizes a different API command structure but will be migrating in the future months)*

**Available capacity zones and machine types:**

| TPU Family | Zone | GCE Machine Type | Max Slice Size | Recommended Provisioning Model |
| :---- | :---- | :---- | :---- | :---- |
| **v6e** | us-east5-a / us-east5-b | ct6e-standard-4t | 8x16 or smaller | FLEX\_START |
| **v6e** | us-central1-a | ct6e-standard-4t | 8x8 or smaller | FLEX\_START |
| **v6e** | europe-west4-a | ct6e-standard-4t | 8x16 or smaller | FLEX\_START |
| **v6e** | southamerica-west1-a | ct6e-standard-4t | 8x16 or smaller | FLEX\_START |
| **v5p** | us-east5-a | ct5p-hightpu-4t | 128 and smaller | FLEX\_START |
| **v5p** | us-central1-a | ct5p-hightpu-4t | 128 and smaller | FLEX\_START |
| **v5e** | us-west4-a | legacy API only | 2x4 or smaller (Serving)  Between 4x4 and 8x16 (Training) | flex-start (legacy) |
| **v5e** | europe-west4-b | legacy API only | 2x4 or smaller (Serving)  Between 4x4 and 8x16 (Training) | flex-start (legacy) |

## 

## Create a VM Instance or Queued Resource

Select the correct command template below matching your target TPU family and size.

### TPU v6e (us-central1-a / us-east5-a / us-east5-b / europe-west4-a / southamerica-west1-a)

To request a TPU v6e 4-chip VM instance using `FLEX_START`:

```shell
gcloud compute instances create tpu-v6e-vm \
    --zone=us-east5-a \
    --machine-type=ct6e-standard-4t \
    --provisioning-model=FLEX_START \
    --request-valid-for-duration=2h \
    --max-run-duration=4h \
    --instance-termination-action=DELETE \
    --image-project=ubuntu-os-accelerator-images \
    --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
    --maintenance-policy=TERMINATE \
    --metadata=startup-script="echo 'TPU VM Booted'"
```

### TPU v5p (us-central1-a / us-east5-a)

To request a TPU v5p 4-chip VM instance using `FLEX_START`:

```shell
gcloud compute instances create tpu-v5p-vm \
    --zone=us-east5-a \
    --machine-type=ct5p-hightpu-4t \
    --provisioning-model=FLEX_START \
    --request-valid-for-duration=2h \
    --max-run-duration=4h \
    --instance-termination-action=DELETE \
    --image-project=ubuntu-os-accelerator-images \
    --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
    --maintenance-policy=TERMINATE \
    --metadata=startup-script="echo 'TPU VM Booted'"
```

### TPU v5e Serving (us-west4-a / europe-west4-b)

*Note: The v5e Alpha API is currently experiencing a backend quota issue that may cause your request to crash with "Code 13: Internal Error" during provisioning. If this happens, please contact support so your project can be manually allowlisted for v5e capacity.*

To request a TPU v5e serving instance, use the Cloud TPU API:

```shell
gcloud alpha compute tpus queued-resources create tpu-v5e-request \
    --zone=us-west4-a \
    --accelerator-type=v5litepod-4 \
    --runtime-version=v2-alpha-tpuv5-lite \
    --node-id=tpu-v5e-node \
    --provisioning-model=flex-start \
    --max-run-duration=4h \
    --valid-until-duration=4h \
    --labels=purpose=flex-start
```

**Check the request status:**

* **For GCE TPU VM Instances (v6e / v5p)**:

```shell
gcloud compute instances describe tpu-v6e-vm --zone=us-east5-a
```

* **For Legacy TPU v5e Queued Resources**:

```shell
gcloud alpha compute tpus queued-resources describe tpu-v5e-request --zone=us-west4-a
```

---

## Setting up a budget warning

To ensure your project does not unintentionally exceed your credit limit, we strongly recommend [configuring a budget alert](https://docs.cloud.google.com/billing/docs/how-to/budgets). 

***Important:** These alerts only send notification emails. They will **NOT** automatically turn off your TPUs or stop charges. You still need to manage idle resources.*

1. In the Cloud Console, open the navigation menu and go to **Billing**  
2. Click on **Budgets & alerts** in the left sidebar, then click **Create budget**  
3. **Name:** Enter a name like "TPU Credit Budget"  
4. **Amount:** Select "Specified amount" and enter **the amount of your credit block**  
5. Under "Manage notifications," set your alert thresholds to **50%**, **75%**, and **90%** of actual spend and save

You will now receive emails when your cost footprint exceeds these thresholds

---

## Automating Setup and Managing Data

Since flex-start VMs do not support pausing/restarting, you may frequently create and delete VMs. To avoid manually setting up your environment or losing data, you should use Persistent Disks and Startup Scripts.

**1\. Create a Persistent Disk** Create a separate disk to hold your data and environments. You only need to run this once. Note: For v5p and v6e TPUs, you must use a `hyperdisk-balanced` or `hyperdisk-ml` disk type.

```shell
gcloud compute disks create my-data-disk \
    --size=100GB \
    --zone=us-east5-a \
    --type=hyperdisk-balanced
```

**2\. Use a Startup Script to Automate Environment Setup** Create a file named `startup.sh` on your local machine with the commands to mount the disk and set up your environment:

Example `startup.sh` might look like:

```shell
#!/bin/bash
# Format the disk only if it does not already contain a filesystem
if [ -z "$(sudo blkid /dev/disk/by-id/google-data-disk)" ]; then
    sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/disk/by-id/google-data-disk
fi
# Mount the disk
sudo mkdir -p /mnt/data
sudo mount -o discard,defaults /dev/disk/by-id/google-data-disk /mnt/data
sudo chmod a+w /mnt/data

# (Optional) Install your libraries automatically
# pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**3\. Attach the Disk and Script when Provisioning** Pass the disk and the script to the VM creation command using `--disk` and `--metadata-from-file`:

**Notes on naming:**

1. `name=my-data-disk` must match the name of the disk you created in Step 1\.  
2. `device-name=data-disk` dictates the path inside the VM (`/dev/disk/by-id/google-data-disk`) used in the `startup.sh` script in Step 2\. If you change this device name, you **must** update the paths in your script to match `google-<your-device-name>`.

```shell
gcloud compute instances create my-tpu-vm \
    --zone=us-east5-a \
    --machine-type=ct6e-standard-4t \
    --provisioning-model=FLEX_START \
    --request-valid-for-duration=2h \
    --max-run-duration=4h \
    --instance-termination-action=DELETE \
    --image-project=ubuntu-os-accelerator-images \
    --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
    --maintenance-policy=TERMINATE \
    --disk=name=my-data-disk,device-name=data-disk,mode=rw,boot=no \
    --metadata-from-file=startup-script=startup.sh
```

##  

## Understanding TPU Quotas

Before manually checking quotas, we highly recommend running the [**Pre-Flight Diagnostics Script**](#run-pre-flight-diagnostics-script-\(recommended\)). The resulting **`tpu_diagnostic_report.txt`** file will explicitly list your current active limits across all regions, removing any guesswork\!

| *Important: The “`GPUS_ALL_REGIONS”` Block* If your diagnostic script or the [Cloud Console (click here to check)](https://console.cloud.google.com/iam-admin/quotas?metric=compute.googleapis.com%2Fgpus_all_regions) shows that your `GPUS_ALL_REGIONS` quota is exactly 0, your entire project is globally restricted from using any accelerators (GPUs or TPUs). This is a standard cloud security measure for new billing accounts. If you see this block, do not try to manually request TPU quota yet. Instead please contact support so the global restriction can be cleared first. |
| :---- |

When requesting TPUs, you may encounter quota limits. Baseline access is granted based on your account reputation, but depending on the region and the API you are using, you might need to manually request an increase in the Cloud Console.

If your diagnostic report shows a quota of 0 for your desired region, you will need to request an increase. The process depends entirely on which architecture you are trying to use:

### For v6e and v5p architectures:

These modern architectures rely on the standard Compute Engine (GCE) quota backend. You can manually request quota increases for these directly through the Cloud Console:

* **v5p Quota Link**: [Click here to check/request v5p quota](https://console.cloud.google.com/iam-admin/quotas?metric=compute.googleapis.com%2Fpreemptible_tpu_v5p) (Metric: ***`PREEMPTIBLE_TPU_V5P`***)  
* **v6e Quota Link**: [Click here to check/request v6e quota](https://console.cloud.google.com/iam-admin/quotas?metric=compute.googleapis.com%2Fpreemptible_tpu_v6e) (Metric: ***`PREEMPTIBLE_TPU_V6E`***)

**Steps to request:**

1. Click the specific link for your TPU generation above (ensure your project is selected at the top of the Cloud Console).  
2. Check the box next to your desired Region (e.g., `us-central1`).  
3. Click **Edit Quotas** at the top right of the page.  
4. Enter the number of chips you need (e.g., `16`). In the **“Request description”** field, briefly describe your workload to help the capacity team identify and prioritize your request, then click submit.  
5. *Note: Quota requests may take up to 2 business days for review. If you receive an immediate automated rejection, it is likely because your billing account is brand new. Please wait 48 hours and try again, or email support.*

### For v5e architectures:

*Note: Because v5e requires the legacy Cloud TPU API, its quota is completely disconnected from Compute Engine. However, you can still manually request quota increases for it through the Cloud Console now that you know the exact preemptible metrics.*

* **v5e Serving Quota Link**: [Click here to request v5e serving quota](https://console.cloud.google.com/iam-admin/quotas?metric=tpu.googleapis.com%2Ftpu-v5s-litepod-serving-preemptible) (Metric: ***`tpu-v5s-litepod-serving-preemptible`***)  
* **v5e Training Quota Link**: [Click here to request v5e training quota](https://console.cloud.google.com/iam-admin/quotas?metric=tpu.googleapis.com%2Ftpu-v5s-litepod-preemptible) (Metric: ***`tpu-v5s-litepod-preemptible`***)

## Recommended frameworks and tutorials

To help you get started as quickly as possible, we have compiled a list of recommended frameworks and their primary use cases on TPUs. Whether you are writing custom ML code from scratch, scaling inference, or fine-tuning existing models, these tools offer the best developer experience for Cloud TPUs.

| Framework / Tool | Best For... | Key Resources & Tutorials |
| :---- | :---- | :---- |
| **Cloud TPU Docs** | A general introduction  | [https://docs.cloud.google.com/tpu/docs/intro-to-tpu](https://docs.cloud.google.com/tpu/docs/intro-to-tpu) |
| **Flex-start VMs** | A flexible and cost-effective way to access TPU resources for AI workloads | • [Request TPU Flex-start VMs](https://docs.cloud.google.com/tpu/docs/request-using-flex-start) |
| **JAX** | Writing high-performance ML code and custom architectures | • [Core JAX Documentation](https://docs.jax.dev/en/latest/) • [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) • [Fine-tune a LLM using TPUs on GKE with JAX](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/train-llm-tpus-gke-jax)  |
| **Tunix** | Scalable and highly efficient LLM post-training on TPUs | • [Tunix Documentation](https://tunix.readthedocs.io/en/latest/index.html) \* [Finetune FunctionGemma 270M for Mobile Actions using Tunix](https://github.com/google-gemini/gemma-cookbook/blob/main/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Tunix.ipynb) See also this recent [Kaggle hackathon](https://www.kaggle.com/competitions/google-tunix-hackathon/overview) to teach a model to reason |
| **Kinetic** | Simple quickstart for deploying code to TPUs | [https://github.com/keras-team/kinetic](https://github.com/keras-team/kinetic) [https://kinetic.readthedocs.io/en/latest/getting\_started.html](https://kinetic.readthedocs.io/en/latest/getting_started.html) |
| **vLLM TPU** | High-throughput, memory-efficient serving of large language and multimodal models | • [vLLM TPU Project Documentation](https://docs.vllm.ai/projects/tpu/en/latest/) • [Recommended Models & Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/) (support matrix incl. quantization formats) • [vLLM Recipes](https://recipes.vllm.ai) (per-model deployment guides) • [tpu-inference on GitHub](https://github.com/vllm-project/tpu-inference) ([releases/changelog](https://github.com/vllm-project/tpu-inference/releases)) |
| **MaxText** | Training and fine tuning highly scalable, high-performance open-source LLMs written in pure JAX | [https://patricktoulme.substack.com/p/frontier-pretraining-infrastructure](https://patricktoulme.substack.com/p/frontier-pretraining-infrastructure) [Supervised fine tuning on a single host with MaxText](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/sft.html) |

## Get support

Expect bugs and rough edges as you get started. If you face any blockers, contact support with your Project ID.

---

##  Troubleshooting & FAQ

This section compiles the most common questions and issues encountered during onboarding.

**Q: Why is my request stuck in `WAITING_FOR_RESOURCES` / `PROVISIONING` indefinitely or failing with `STOCKOUT` / `RESOURCE_POOL_EXHAUSTED`?**

**A:** New cloud projects default to a limit of 0.0 for the GPUS\_ALL\_REGIONS (Global Accelerator Quota) metric. Even if your regional TPU quota shows active allocation, this global limit blocks the platform from actually spinning up your instance. This is an anti-fraud safeguard. To fix this:

1. **Link a Billing Account**: Go to the Cloud Console \-\> Billing and ensure your project has an active, linked paid billing account (credits/coupons alone are not enough if billing is disabled).  
2. **Increase Billing Reputation (Credit Card/Bank Link)**: To verify your identity and automatically lift anti-fraud limits, link a valid credit card or bank account to your billing profile in the Console. (This is for identity verification and will *not* charge your card if you are funded by credits).  
3. **Institutional / Invoice Override fallback**: If you are using a university-managed invoice account and cannot associate a personal card, you must contact support with your Billing Account ID so a manual reputation whitelist can be requested from the billing backend.  
4. **Request Quota Increase**: Once billing is verified, go to IAM & Admin \> Quotas in the Cloud Console, search for `GPUS_ALL_REGIONS`, click Edit Quotas, and request an increase to 8 or 16\.  
5. **Resubmit Request**: Once approved, delete your stuck instance (`gcloud compute instances delete <VM_NAME> --zone=<ZONE> --quiet` or the legacy queued-resource command `gcloud alpha compute tpus queued-resources delete <QR_NAME> --zone=<ZONE> --force`) and submit your request again.

**Q: Why am I getting `Error: code 10, Failed to perform tenant project creation` when I submit my TPU request?**  
**A:** This typically happens if you are reusing a legacy cloud project that was previously associated with other programs (such as the legacy TPU Research Credits program), or if a prior setup was deleted incorrectly. To resolve this, **always create a completely net-new, isolated cloud project**.

**Q: Why did my TPU instance or training job suddenly terminate after exactly 24 hours?**  
**A:** By default, Flex-Start VM instances are set to terminate after 24 hours. To run your training jobs for up to 7 days, append the `--max-run-duration=168h` and `--instance-termination-action=DELETE` flags to your creation command:

```shell
gcloud compute instances create my-tpu-vm \
    --zone=us-east5-a \
    --machine-type=ct6e-standard-4t \
    --provisioning-model=FLEX_START \
    --request-valid-for-duration=4h \
    --max-run-duration=168h \
    --instance-termination-action=DELETE \
    --image-project=ubuntu-os-accelerator-images \
    --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
    --maintenance-policy=TERMINATE
```

**Q: My dataset is larger than the 100 GiB boot disk. How do I allocate more storage?**  
**A:** TPU VMs ship with a non-resizable 100 GiB system disk. To manage larger datasets, you should either stream them from a **Cloud Storage (GCS) bucket** or attach a dedicated **Hyperdisk ML** volume. If using GCS, make sure to grant the storage admin role to your default compute service account:

```shell
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET_NAME \
    --member=serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com \
    --role=roles/storage.objectAdmin
```

**Q: Can I use Colab Enterprise or other browser-based notebook interfaces with my TPU?**  
**A:** Colab Enterprise does not natively support self-managed Cloud TPU VMs. Instead, we strongly recommend connecting your local **Antigravity** to the TPU via **Remote-SSH** and launching standard Jupyter Notebooks over an SSH port tunnel.

**Q: How do I stop a Flex-start VM early? Can I pause and restart it?**  
**A:** Currently, TPU VMs created with FLEX\_START or the standard Compute Engine API do not support the stop and restart operations. If you are finished with your instance early, or need to pause, you must delete the instance. To preserve your environment and data across instances, we strongly recommend automating your setup using a Startup Script or keeping data on a separate Persistent Disk.

---

## Resources

* JAX Resources \- [https://github.com/rcrowe-google/Learning-JAX](https://github.com/rcrowe-google/Learning-JAX)  
* Introduction to TPU \- [https://jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/)  
* vLLM Recipes (per-model deployment guides) \- [https://recipes.vllm.ai](https://recipes.vllm.ai)  
* vLLM TPU backend on GitHub \- [https://github.com/vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)

# Creating a TPU via Console

## Visual Walkthrough: Requesting a TPU via Cloud Console

If you prefer using the Cloud Console over the `gcloud` CLI, you can now provision TPUs and use the Flex-start queue directly from the UI. Follow these steps to request and verify your TPU.

### Step 1: Navigate to Compute Engine

From your Cloud Console dashboard, navigate to the [**Compute Engine \-\> VM instances**](https://console.cloud.google.com/compute/instances) page using the search bar or left-hand navigation menu.

![][image1]

From the Compute Engine dashboard, click the blue **Create instance** button (or jump straight there using the [**Create an instance link**](https://console.cloud.google.com/compute/instancesAdd)).  
![][image2]

Step 2: Configure Your Machine Type  
![][image3]

Give your instance a name and select your desired Region and Zone (e.g., `us-east5-a` for v6e TPUs).  
![][image4]

Under the **Machine configuration** section, select the **TPUs** tab. Choose your preferred TPU generation (like the `CT6E` series). And select your specific **Machine type** topology from the dropdown (e.g., `ct6e-standard-1t` for a single-chip instance).  
![][image5]

### Step 3: Enable Flex-start Provisioning

Under the **VM provisioning model** dropdown, select **Flex-start**. This ensures your request enters the queue and is automatically provisioned as soon as capacity becomes available.![][image6]

Click **Create** at the bottom of the page.

### Step 4: Connect to Your Instance

Your instance will appear in your [**VM instances list**](https://console.cloud.google.com/compute/instances) with a loading spinner while it waits in the Flex-start queue.

![][image7]

Once capacity is secured, the status will change to a green checkmark. Click the **SSH** button to launch a secure, browser-based terminal directly into your TPU VM.

![][image8]

###  Step 5: Verify Your Hardware

Once your SSH session connects, you are inside your TPU VM\!![][image9]

To prove your TPU is attached and ready for machine learning workloads, set up a Python virtual environment and install JAX. Run the following commands to check your device count:

```shell
# Set up environment and install JAX
sudo apt update && sudo apt install -y python3-pip python3-venv
python3 -m venv tpu-env
source tpu-env/bin/activate
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Run the JAX device count check
python3 -c "import jax; print(f'🎉 TPU Cores Found: {jax.device_count()}')"
```

If it prints the number of cores you requested, your TPU is successfully online and ready for you to start building\!  
![][image10]

### Stop or Delete your instance.

When you are done with your instance you can select it and stop or delete from the menu to make sure you are no longer charged.  
![][image11]  
