#!/bin/bash
# Google Cloud setup for the Gemma 4 DevOps agents: verifies auth, resolves the
# project ID, installs Python dependencies, enables APIs, and grants IAM roles
# to the default compute service account. Safe to rerun.
#
# Run with ./init.sh. Sourcing also works and is only needed in Cloud Shell,
# where it adds ~/.local/bin to the current shell's PATH.

main() {
    local PROJECT_FILE="$HOME/project_id.txt"
    local SCRIPT_DIR ERRORS=0
    SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

    if [ -n "$CLOUD_SHELL" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # --- Authentication ---
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        echo "Error: no active gcloud account. Run 'gcloud auth login' and retry." >&2
        return 1
    fi

    if [ -z "$CLOUD_SHELL" ]; then
        if gcloud auth application-default print-access-token >/dev/null 2>&1; then
            echo "ADC is valid."
        else
            echo "ADC expired or not found. Initializing login..."
            gcloud auth application-default login || return 1
        fi
    fi

    # --- Project ID: reuse the saved one if valid, otherwise prompt ---
    local PROJECT_ID=""
    if [ -s "$PROJECT_FILE" ]; then
        PROJECT_ID=$(tr -d '[:space:]' < "$PROJECT_FILE")
        echo "Found saved project ID in $PROJECT_FILE: $PROJECT_ID"
        if ! gcloud projects describe "$PROJECT_ID" --quiet >/dev/null 2>&1; then
            echo "Warning: project '$PROJECT_ID' does not exist or you lack access to it." >&2
            rm "$PROJECT_FILE"
            PROJECT_ID=""
        fi
    fi
    while [ -z "$PROJECT_ID" ]; do
        if ! read -r -p "Enter project ID: " PROJECT_ID; then
            echo "Error: no project ID provided and none saved in $PROJECT_FILE." >&2
            return 1
        fi
        if ! gcloud projects describe "$PROJECT_ID" --quiet >/dev/null 2>&1; then
            echo "Project '$PROJECT_ID' does not exist or you lack access to it. Try again." >&2
            PROJECT_ID=""
        fi
    done
    echo "$PROJECT_ID" > "$PROJECT_FILE"

    gcloud config set project "$PROJECT_ID" --quiet || return 1
    if [ -z "$CLOUD_SHELL" ]; then
        gcloud auth application-default set-quota-project "$PROJECT_ID" --quiet >/dev/null 2>&1
    fi
    echo "Active project: $PROJECT_ID"

    echo -e "\n--- Installing Python dependencies ---"
    if ! python3 -m pip install -q -r "$SCRIPT_DIR/requirements.txt"; then
        echo "Warning: pip install failed." >&2
        ERRORS=$((ERRORS + 1))
    fi

    echo -e "\n--- Enabling APIs ---"
    if ! gcloud services enable compute.googleapis.com \
                                artifactregistry.googleapis.com \
                                run.googleapis.com \
                                cloudbuild.googleapis.com \
                                iam.googleapis.com \
                                aiplatform.googleapis.com \
                                tpu.googleapis.com \
                                secretmanager.googleapis.com; then
        echo "Warning: failed to enable one or more APIs." >&2
        ERRORS=$((ERRORS + 1))
    fi

    echo -e "\n--- Granting IAM roles to the default compute service account ---"
    local PROJECT_NUMBER
    PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
    if [ -z "$PROJECT_NUMBER" ]; then
        echo "Error: could not determine the project number for '$PROJECT_ID'." >&2
        return 1
    fi
    local SA_EMAIL="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    echo "Service account: $SA_EMAIL"

    local EXISTING_ROLES
    EXISTING_ROLES=$(gcloud projects get-iam-policy "$PROJECT_ID" \
        --flatten="bindings[].members" \
        --filter="bindings.members:serviceAccount:$SA_EMAIL" \
        --format="value(bindings.role)")

    local ROLES=(
        "roles/logging.logWriter"
        "roles/logging.viewer"
        "roles/monitoring.metricWriter"
        "roles/stackdriver.resourceMetadata.writer"
        "roles/tpu.admin"
        "roles/secretmanager.secretAccessor"
        "roles/iam.serviceAccountUser"
        "roles/compute.instanceAdmin.v1"
        "roles/artifactregistry.reader"
    )
    local ROLE
    for ROLE in "${ROLES[@]}"; do
        if grep -qx "$ROLE" <<< "$EXISTING_ROLES"; then
            echo "$ROLE already granted."
            continue
        fi
        echo "Adding $ROLE..."
        if ! gcloud projects add-iam-policy-binding "$PROJECT_ID" \
                --member="serviceAccount:$SA_EMAIL" \
                --role="$ROLE" \
                --condition=None \
                --quiet >/dev/null; then
            echo "Warning: failed to add $ROLE." >&2
            ERRORS=$((ERRORS + 1))
        fi
    done

    # --- Register the tpu-devops MCP server for this repo ---
    # .mcp.json is generated (and gitignored) because it embeds the GCP project id;
    # project-setup.sh owns the writing logic, so delegate to it. Only generate when
    # the entry is missing — project-setup.sh rewrites it with default model settings,
    # which would clobber a customized one (rerun it directly with flags to regenerate).
    echo -e "\n--- Registering the tpu-devops MCP server (.mcp.json) ---"
    if [ -f "$SCRIPT_DIR/.mcp.json" ] && grep -q '"tpu-devops"' "$SCRIPT_DIR/.mcp.json"; then
        echo "tpu-devops already registered in .mcp.json; leaving it untouched."
        echo "To regenerate: ./project-setup.sh . --project $PROJECT_ID [--model ... --accelerator ... --tp ...]"
    elif [ -f "$SCRIPT_DIR/project-setup.sh" ]; then
        if ! "$SCRIPT_DIR/project-setup.sh" "$SCRIPT_DIR" --project "$PROJECT_ID" --skip-deps; then
            echo "Warning: MCP server registration failed." >&2
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo "Warning: project-setup.sh not found next to init.sh; skipping MCP registration." >&2
        ERRORS=$((ERRORS + 1))
    fi

    echo
    if [ "$ERRORS" -eq 0 ]; then
        echo "--- Full setup complete ---"
    else
        echo "--- Setup finished with $ERRORS error(s); see warnings above ---" >&2
        return 1
    fi
}

main "$@"
