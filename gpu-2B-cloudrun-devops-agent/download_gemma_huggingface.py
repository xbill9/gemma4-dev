import argparse
import os

from huggingface_hub import snapshot_download


def download_gemma(repo_id: str = "google/gemma-4-E2B-it"):
    """Downloads Gemma weights using huggingface_hub."""
    print(f"🚀 Downloading {repo_id} from Hugging Face...")

    # Retrieve the Hugging Face token from environment or local cache
    token = os.getenv("HF_TOKEN")
    if not token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            try:
                with open(token_path, "r") as f:
                    token = f.read().strip()
            except Exception as e:
                print(f"⚠️ Failed to read Hugging Face token from cache: {e}")

    try:
        path = snapshot_download(repo_id=repo_id, token=token)
        print(f"✅ Model downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Failed to download model from Hugging Face: {e}")
        print("Please ensure your HF_TOKEN is set or you have run 'huggingface-cli login'.")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Gemma weights via huggingface_hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="google/gemma-4-E2B-it",
        help="Hugging Face model repo ID",
    )
    args = parser.parse_args()

    download_gemma(args.repo_id)
