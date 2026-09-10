def get_model_artifact_uri(project_id, publisher="google", model="gemma-4-E2B-it"):
    from google.cloud.aiplatform_v1.services.model_garden_service import (
        ModelGardenServiceClient,
    )

    client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    client = ModelGardenServiceClient(client_options=client_options)

    # Format the publisher model name
    name = f"publishers/{publisher}/models/{model}"

    try:
        response = client.get_publisher_model(name=name)
        print(f"\n[Model Found: {response.name}]")

        # Check for artifact URIs in supported_actions.deploy
        if response.supported_actions and response.supported_actions.deploy:
            deploy_action = response.supported_actions.deploy
            print("\n--- Deployment Artifacts ---")
            if deploy_action.artifact_uri:
                print(f"Artifact URI: {deploy_action.artifact_uri}")
            if deploy_action.public_artifact_uri:
                from urllib.parse import unquote

                decoded_uri = unquote(deploy_action.public_artifact_uri)
                print(f"Public Artifact URI (Decoded):\n{decoded_uri}")
            if deploy_action.container_spec:
                print(f"Container Spec Image: {deploy_action.container_spec.image_uri}")

        # Check for notebooks
        if response.supported_actions and response.supported_actions.open_notebooks:
            print("\n--- Notebooks ---")
            for notebook in response.supported_actions.open_notebooks.notebooks:
                print(f"Title: {notebook.title}")
                for region, ref in notebook.references.items():
                    print(f"  [{region}] {ref.resource_name or ref.uri}")

        # Check for GKE deployment configs
        if response.supported_actions and response.supported_actions.deploy_gke:
            print("\n--- GKE Configs ---")
            for config in response.supported_actions.deploy_gke.gke_yaml_configs:
                # Print just a snippet if it's long
                print(f"YAML Config Snippet: {config[:200]}...")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import os

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "aisprint-491218")
    get_model_artifact_uri(project_id, model="gemma-4@gemma-4-E2B-it")
