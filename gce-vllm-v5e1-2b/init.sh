#!/bin/bash

# --- Function for error handling ---
handle_error() {
  echo -e "\n\n*******************************************************"
  echo "Error: $1"
  echo "*******************************************************"
  # Instead of exiting, we warn the user and wait for input
  echo "The script encountered an error."
  echo "Press [Enter] to ignore this error and attempt to continue."
  echo "Press [Ctrl+C] to exit the script completely."
  read -r # Pauses script here
}

# Add $HOME/.local/bin to PATH if running in Cloud Shell
if [ -n "$CLOUD_SHELL" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "Error: No active gcloud account found."
    echo "Please run 'gcloud auth login' and try again."
    exit 1
fi

if [ -z "$CLOUD_SHELL" ]; then
    if ! gcloud auth application-default print-access-token > /dev/null 2>&1; then
        echo "ADC expired or not found. Initializing login..."
        gcloud auth application-default login
    else
        echo "ADC is valid."
    fi
fi


# --- Part 1: Find or Create Google Cloud Project ID ---
PROJECT_FILE="$HOME/project_id.txt"
PROJECT_ID_SET=false

# Check if a project ID file already exists and points to a valid project
if [ -s "$PROJECT_FILE" ]; then
    EXISTING_PROJECT_ID=$(cat "$PROJECT_FILE" | tr -d '[:space:]') # Read and trim whitespace
    echo "--- Found existing project ID in $PROJECT_FILE: $EXISTING_PROJECT_ID ---"
    echo "Verifying this project exists in Google Cloud..."

    # Check if the project actually exists in GCP and we have permission to see it
    if gcloud projects describe "$EXISTING_PROJECT_ID" --quiet >/dev/null 2>&1; then
        echo "Project '$EXISTING_PROJECT_ID' successfully verified."
        FINAL_PROJECT_ID=$EXISTING_PROJECT_ID
        PROJECT_ID_SET=true

        # Ensure gcloud config is set to this project for the current session
        gcloud config set project "$FINAL_PROJECT_ID" || handle_error "Failed to set active project to '$FINAL_PROJECT_ID'."
        echo "Set active gcloud project to '$FINAL_PROJECT_ID'."
    else
        echo "Warning: Project '$EXISTING_PROJECT_ID' from file does not exist or you lack permissions."
        echo "Removing invalid reference file and proceeding with new project creation."
        rm "$PROJECT_FILE"
    fi
else
    read -p "Enter Project ID: " PROJECT_ID
    echo "$PROJECT_ID" > "$HOME/project_id.txt"	
fi


# --- Part 2: Install Dependencies and Run Billing Setup ---
echo -e "\n--- Installing Python dependencies ---"
# Using || handle_error means if it fails, it will pause, allow you to read, and then proceed
pip install -r requirements.txt

PROJECT_ID=$(cat "$PROJECT_FILE")
echo -e "\n--- set Project id to $PROJECT_ID ---"
gcloud config set project "$PROJECT_ID"

echo -e "\n--- Enable APIs ---"
gcloud services enable  compute.googleapis.com \
                        artifactregistry.googleapis.com \
                        run.googleapis.com \
                        cloudbuild.googleapis.com \
                        iam.googleapis.com \
                        aiplatform.googleapis.com \
                        tpu.googleapis.com \
                        secretmanager.googleapis.com

echo -e "\n--- Grant IAM Permissions to Service Account ---"
# Get the project number to construct the default compute service account
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
DEFAULT_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
SA_EMAIL=$DEFAULT_SA
echo "Setting permissions for default service account: $SA_EMAIL"

if [ -n "$SA_EMAIL" ]; then
    echo "Granting roles to $SA_EMAIL..."
    ROLES=(
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

    for ROLE in "${ROLES[@]}"; do
        echo "Adding $ROLE..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$SA_EMAIL" \
            --role="$ROLE" \
            --quiet > /dev/null || echo "Failed to add $ROLE"
    done
else
    echo "No service account provided. Skipping IAM grants."
fi

echo -e "\n--- Full Setup Complete ---"


