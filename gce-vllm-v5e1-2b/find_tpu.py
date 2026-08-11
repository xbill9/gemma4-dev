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


def find_tpu_vms():
    print("🔍 Fetching Compute Engine TPU VMs...")
    stdout = run_gcloud(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--project={PROJECT_ID}",
            "--zone=-",
            "--format=json(name,zone,acceleratorType,topology,status)",
        ]
    )
    if not stdout:
        return []

    try:
        tpus = json.loads(stdout)
        tpu_vms = []
        for tpu in tpus:
            zone = tpu.get("zone", "").split("/")[-1]
            tpu_vms.append(
                {
                    "name": tpu.get("name"),
                    "zone": zone,
                    "accelerator_type": tpu.get("acceleratorType"),
                    "topology": tpu.get("topology"),
                    "status": tpu.get("status"),
                }
            )
        return tpu_vms
    except Exception as e:
        print(f"Error parsing GCE TPU VMs: {e}", file=sys.stderr)
        return []


def find_tpu_queued_resources():
    print("🔍 Fetching TPU Queued Resources...")
    stdout = run_gcloud(
        [
            "alpha",
            "compute",
            "tpus",
            "queued-resources",
            "list",
            f"--project={PROJECT_ID}",
            "--zone=-",
            "--format=json(name,zone,acceleratorType,topology,state)",
        ]
    )
    if not stdout:
        return []

    try:
        resources = json.loads(stdout)
        tpu_qrs = []
        for qr in resources:
            zone = qr.get("zone", "").split("/")[-1]
            tpu_qrs.append(
                {
                    "name": qr.get("name", "").split("/")[-1],
                    "zone": zone,
                    "accelerator_type": qr.get("acceleratorType"),
                    "topology": qr.get("topology"),
                    "state": qr.get("state", {}).get("state", "UNKNOWN"),
                }
            )
        return tpu_qrs
    except Exception as e:
        print(f"Error parsing TPU Queued Resources: {e}", file=sys.stderr)
        return []


def find_gce_tpu_instances():
    """TPU-bearing Compute Engine instances — the namespace THIS rig provisions into.

    `find_tpu_vms()` above lists the Cloud TPU API's node namespace, and a `ct5lp-*` instance
    created by `gcloud compute instances create` does not appear there at all. Without this
    section the report would show none of this rig's own resources while looking complete.
    """
    print("🔍 Fetching TPU-bearing Compute Engine instances...")
    stdout = run_gcloud(
        [
            "compute",
            "instances",
            "list",
            f"--project={PROJECT_ID}",
            "--filter=machineType~'ct5lp|ct5p|ct6e'",
            "--format=json(name,zone,machineType,status,labels,scheduling)",
        ]
    )
    if not stdout:
        return []

    try:
        instances = json.loads(stdout)
        rows = []
        for inst in instances:
            rows.append(
                {
                    "name": inst.get("name"),
                    "zone": (inst.get("zone") or "").split("/")[-1],
                    "machine_type": (inst.get("machineType") or "").split("/")[-1],
                    "status": inst.get("status"),
                    "rig": (inst.get("labels") or {}).get("rig", "—"),
                    "model": (inst.get("scheduling") or {}).get("provisioningModel", "—"),
                }
            )
        return rows
    except Exception as e:
        print(f"Error parsing Compute Engine TPU instances: {e}", file=sys.stderr)
        return []


def find_tpu_quotas():
    print("🔍 Fetching available TPU v5e zone quotas...")
    stdout = run_gcloud(
        [
            "beta",
            "quotas",
            "info",
            "list",
            "--service=tpu.googleapis.com",
            f"--project={PROJECT_ID}",
            "--filter=quotaId:TPUV5sLitepodPerProjectPerZoneForTPUAPI",
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
        print(f"Error parsing TPU quotas: {e}", file=sys.stderr)
        return []


def main():
    gce_instances = find_gce_tpu_instances()
    tpu_vms = find_tpu_vms()
    tpu_qrs = find_tpu_queued_resources()
    tpu_quotas = find_tpu_quotas()

    report = []
    report.append("# 🚀 GCP TPU Resource Discovery Report")
    report.append(f"**Project:** `{PROJECT_ID}`\n")
    report.append(
        "This report spans **both control planes**. The first section is the one this rig "
        "provisions into; the two after it belong to the Cloud TPU API and are shown because "
        "sibling rigs compete for the same physical chips.\n"
    )

    report.append("## 🧩 TPU Instances on Compute Engine (this rig's namespace)")
    if gce_instances:
        report.append("| Instance Name | Zone | Machine Type | Status | Rig | Provisioning |")
        report.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for inst in gce_instances:
            report.append(
                f"| **{inst['name']}** | `{inst['zone']}` | `{inst['machine_type']}` | "
                f"`{inst['status']}` | `{inst['rig']}` | `{inst['model']}` |"
            )
    else:
        report.append("_No TPU-bearing Compute Engine instances found in the project._")
    report.append("")

    report.append("## 🖥️ Cloud TPU API VM Nodes (other control plane)")
    if tpu_vms:
        report.append("| Instance Name | Zone | Accelerator Type | Topology | Status |")
        report.append("| :--- | :--- | :--- | :--- | :--- |")
        for tpu in tpu_vms:
            report.append(
                f"| **{tpu['name']}** | `{tpu['zone']}` | `{tpu['accelerator_type']}` | `{tpu['topology']}` | `{tpu['status']}` |"
            )
    else:
        report.append("_No TPU VM instances found in the project._")
    report.append("")

    report.append("## 🐳 TPU Queued Resources (Flex-start)")
    if tpu_qrs:
        report.append("| Resource ID | Zone | Accelerator Type | Topology | State |")
        report.append("| :--- | :--- | :--- | :--- | :--- |")
        for qr in tpu_qrs:
            report.append(
                f"| **{qr['name']}** | `{qr['zone']}` | `{qr['accelerator_type']}` | `{qr['topology']}` | `{qr['state']}` |"
            )
    else:
        report.append("_No TPU Queued Resources found in the project._")
    report.append("")

    report.append("## 📊 Available TPU v5e Quotas")
    if tpu_quotas:
        report.append("Below are the zones where the project has non-zero quota limits for TPU v5e:")
        report.append("")
        report.append("| Zone | Limit (Value) |")
        report.append("| :--- | :--- |")
        for zone, limit in tpu_quotas:
            limit_display = "Default (-1)" if limit == "-1" else limit
            report.append(f"| `{zone}` | {limit_display} |")
    else:
        report.append("_No zones found with available TPU v5e quota._")

    report_text = "\n".join(report)
    print("\n" + "=" * 40 + "\n")
    print(report_text)

    # Save the report as an artifact
    # Defaults to the CWD; ARTIFACT_DIR redirects it. This was a hardcoded UUID path inside
    # another tool's scratch directory, so the report only landed on one machine.
    artifact_dir = os.getenv("ARTIFACT_DIR", ".")
    os.makedirs(artifact_dir, exist_ok=True)
    report_path = os.path.join(artifact_dir, "tpu_discovery_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n💾 Report saved as artifact: [tpu_discovery_report.md](file://{report_path})")


if __name__ == "__main__":
    main()
