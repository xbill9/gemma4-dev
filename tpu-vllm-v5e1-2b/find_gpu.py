#!/usr/bin/env python3
import json
import os
import subprocess
import sys

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aisprint-491218")


def run_gcloud(args):
    cmd = ["gcloud"] + args
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running {' '.join(cmd)}: {e.stderr}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def find_gpu_vms():
    print("🔍 Fetching Compute Engine instances with GPUs...")
    stdout = run_gcloud(
        [
            "compute",
            "instances",
            "list",
            f"--project={PROJECT_ID}",
            "--format=json(name,zone,machineType,status,guestAccelerators)",
        ]
    )
    if not stdout:
        return []

    try:
        instances = json.loads(stdout)
        gpu_vms = []
        for inst in instances:
            guest_acc = inst.get("guestAccelerators", [])
            machine_type = inst.get("machineType", "")
            # Identify GPU if guestAccelerators is present, or machine type has g2/a2/a3
            is_gpu = len(guest_acc) > 0 or any(x in machine_type.lower() for x in ["g2-", "a2-", "a3-"])
            if is_gpu:
                zone = inst.get("zone", "").split("/")[-1]
                mtype = machine_type.split("/")[-1]
                acc_info = []
                for acc in guest_acc:
                    acc_type = acc.get("acceleratorType", "").split("/")[-1]
                    acc_count = acc.get("acceleratorCount", 1)
                    acc_info.append(f"{acc_count}x {acc_type}")
                acc_str = ", ".join(acc_info) if acc_info else "Yes (G2/A2/A3 VM)"
                gpu_vms.append(
                    {
                        "name": inst.get("name"),
                        "zone": zone,
                        "machine_type": mtype,
                        "status": inst.get("status"),
                        "accelerators": acc_str,
                    }
                )
        return gpu_vms
    except Exception as e:
        print(f"Error parsing GCE VMs: {e}", file=sys.stderr)
        return []


def find_cloud_run_gpu_services():
    print("🔍 Fetching Cloud Run services...")
    stdout = run_gcloud(
        [
            "run",
            "services",
            "list",
            f"--project={PROJECT_ID}",
            "--format=json(metadata.name,metadata.labels,status.address.url,spec.template.spec.containers)",
        ]
    )
    if not stdout:
        return []

    try:
        services = json.loads(stdout)
        gpu_services = []
        for svc in services:
            metadata = svc.get("metadata", {})
            name = metadata.get("name", "")
            status = svc.get("status", {})
            url = status.get("address", {}).get("url", "")

            # Check containers resource limits for GPU
            spec = svc.get("spec", {})
            containers = spec.get("template", {}).get("spec", {}).get("containers", [])
            has_gpu = False
            gpu_type = "nvidia-l4"  # Default for Cloud Run GPU
            gpu_count = 0

            for container in containers:
                resources = container.get("resources", {})
                limits = resources.get("limits", {})
                if "run.googleapis.com/gpu" in limits:
                    has_gpu = True
                    gpu_count = limits["run.googleapis.com/gpu"]

            # Alternatively check if name starts with "gpu-" as a backup indicator
            if has_gpu or name.startswith("gpu-"):
                gpu_services.append(
                    {
                        "name": name,
                        "url": url,
                        "gpus": f"{gpu_count}x {gpu_type}" if gpu_count else "1x nvidia-l4 (Estimated)",
                    }
                )
        return gpu_services
    except Exception as e:
        print(f"Error parsing Cloud Run services: {e}", file=sys.stderr)
        return []


def find_gpu_quotas():
    print("🔍 Fetching available GPU zone quotas (NVIDIA-L4-GPUS-per-project-zone)...")
    stdout = run_gcloud(
        [
            "beta",
            "quotas",
            "info",
            "list",
            "--service=compute.googleapis.com",
            f"--project={PROJECT_ID}",
            "--filter=quotaId:NVIDIA-L4-GPUS-per-project-zone",
            "--format=json",
        ]
    )
    if not stdout:
        return []

    try:
        quota_data = json.loads(stdout)
        available_zones = []
        for info in quota_data:
            dimensions_infos = info.get("dimensionsInfos", [])
            for dim_info in dimensions_infos:
                details = dim_info.get("details", {})
                limit_val = details.get("value")
                # limit_val can be "-1" (unlimited/default) or a positive number
                if limit_val and limit_val != "0":
                    dim_map = dim_info.get("dimensions", {})
                    zone_val = dim_map.get("zone") or dim_map.get("region")
                    if zone_val:
                        available_zones.append((zone_val, limit_val))
                    else:
                        locations = dim_info.get("applicableLocations", [])
                        for loc in locations:
                            available_zones.append((loc, limit_val))
        return sorted(list(set(available_zones)))
    except Exception as e:
        print(f"Error parsing GPU quotas: {e}", file=sys.stderr)
        return []


def main():
    gpu_vms = find_gpu_vms()
    gpu_services = find_cloud_run_gpu_services()
    gpu_quotas = find_gpu_quotas()

    report = []
    report.append("# 🚀 GCP GPU Resource Discovery Report")
    report.append(f"**Project:** `{PROJECT_ID}`\n")

    report.append("## 🖥️ Compute Engine GPU VMs")
    if gpu_vms:
        report.append("| VM Name | Zone | Machine Type | Status | Accelerator(s) |")
        report.append("| :--- | :--- | :--- | :--- | :--- |")
        for vm in gpu_vms:
            report.append(
                f"| **{vm['name']}** | `{vm['zone']}` | `{vm['machine_type']}` | `{vm['status']}` | {vm['accelerators']} |"
            )
    else:
        report.append("_No GPU VM instances found in the project._")
    report.append("")

    report.append("## 🐳 Cloud Run GPU Services")
    if gpu_services:
        report.append("| Service Name | GPU Configuration | Active Endpoint URL |")
        report.append("| :--- | :--- | :--- |")
        for svc in gpu_services:
            report.append(f"| **{svc['name']}** | `{svc['gpus']}` | [{svc['url']}]({svc['url']}) |")
    else:
        report.append("_No Cloud Run GPU services found in the project._")
    report.append("")

    report.append("## 📊 Available GPU Quotas (nvidia-l4)")
    if gpu_quotas:
        report.append("Below are the zones where the project has non-zero quota limits for NVIDIA L4 GPUs:")
        report.append("")
        report.append("| Zone | Limit (Value) |")
        report.append("| :--- | :--- |")
        for zone, limit in gpu_quotas:
            limit_display = "Default (-1)" if limit == "-1" else limit
            report.append(f"| `{zone}` | {limit_display} |")
    else:
        report.append("_No zones found with available NVIDIA L4 GPU quota._")

    report_text = "\n".join(report)
    print("\n" + "=" * 40 + "\n")
    print(report_text)

    # Save the report as an artifact
    artifact_dir = "/home/xbill/.gemini/antigravity-cli/brain/d7571dec-bde7-4d8b-993e-256bb06732e8"
    os.makedirs(artifact_dir, exist_ok=True)
    report_path = os.path.join(artifact_dir, "gpu_discovery_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n💾 Report saved as artifact: [gpu_discovery_report.md](file://{report_path})")


if __name__ == "__main__":
    main()
