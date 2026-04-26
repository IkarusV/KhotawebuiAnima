"""Upload Anima models to Modal Volume from HuggingFace URLs."""
import modal
import os

app = modal.App("anima-model-uploader")
vol = modal.Volume.from_name("anima-training-data", create_if_missing=True)
VOL_MOUNT = "/data"

DIT_URL = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview3-base.safetensors"
QWEN3_URL = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors"
VAE_URL = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors"


@app.function(volumes={VOL_MOUNT: vol}, timeout=3600)
def download_models():
    """Download base models from HuggingFace directly into the Volume."""
    import urllib.request

    models_dir = f"{VOL_MOUNT}/models"
    os.makedirs(models_dir, exist_ok=True)

    url_map = [
        (DIT_URL, "anima-preview3-base.safetensors"),
        (QWEN3_URL, "qwen_3_06b_base.safetensors"),
        (VAE_URL, "qwen_image_vae.safetensors"),
    ]

    results = []
    for url, fname in url_map:
        dest = f"{models_dir}/{fname}"
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            results.append(f"Already exists: {fname} ({size_mb:.0f} MB)")
            continue
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        results.append(f"Downloaded {fname} ({size_mb:.0f} MB)")
        print(f"  Done: {size_mb:.0f} MB")

    vol.commit()
    return results


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("  Uploading Anima Preview 3 models to Modal Volume")
    print("=" * 60)
    results = download_models.remote()
    print()
    for r in results:
        print(f"  {r}")
    print()
    print("All done! Models are ready on Modal cloud.")
