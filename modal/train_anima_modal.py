"""
Anima LoRA Training on Modal Cloud GPUs.
This script defines a Modal App that:
  1. Downloads sd-scripts from github and installs it
  2. Uploads datasets and base models to a persistent Volume
  3. Trains Anima LoRAs on A100/H100/L4 GPUs
  4. Saves results to the Volume for download
"""

import modal
import os

# ─── Modal Setup ──────────────────────────────────────────────────────────

# GPU can be overridden by setting ANIMA_GPU env var before importing
DEFAULT_GPU = os.environ.get("ANIMA_GPU", "H100")

app = modal.App("anima-lora-trainer")

# Persistent volume for models, datasets, and outputs
vol = modal.Volume.from_name("anima-training-data", create_if_missing=True)
VOL_MOUNT = "/data"

# Container image with sd-scripts and all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "wget")
    # Force clean install of PyTorch + torchvision from CUDA index
    .run_commands(
        "pip uninstall -y torch torchvision xformers 2>/dev/null; "
        "pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124",
        # Verify torch+torchvision work together
        "python -c 'import torch; import torchvision; print(f\"torch={torch.__version__} tv={torchvision.__version__} cuda={torch.cuda.is_available()}\")'",
    )
    .env({"PYTORCH_DISABLE_META_REGISTRATIONS": "1"})
    .pip_install(
        "accelerate==1.6.0",
        "transformers==4.54.1",
        "diffusers[torch]==0.32.1",
        "ftfy==6.3.1",
        "opencv-python==4.10.0.84",
        "einops==0.7.0",
        "bitsandbytes",
        "lion-pytorch==0.2.3",
        "schedulefree==1.4",
        "pytorch-optimizer==3.9.0",
        "prodigy-plus-schedule-free==1.9.2",
        "prodigyopt==1.1.2",
        "safetensors==0.4.5",
        "toml==0.10.2",
        "voluptuous==0.15.2",
        "huggingface-hub==0.34.3",
        "imagesize==1.4.1",
        "numpy",
        "rich==14.1.0",
        "sentencepiece==0.2.1",
        "xformers==0.0.29.post1",
        "tensorboard",
    )
    .run_commands(
        "git clone https://github.com/kohya-ss/sd-scripts.git /opt/sd-scripts",
        "cd /opt/sd-scripts && pip install --no-deps -e .",
        # Final check - cache bust v3
        "python -c 'import torch; from torchvision import transforms; print(\"OK: torchvision works\")'",
    )
)


# ─── Upload Functions ─────────────────────────────────────────────────────

@app.function(volumes={VOL_MOUNT: vol}, timeout=3600)
def upload_models_from_urls(dit_url: str, qwen3_url: str, vae_url: str):
    """Download base models from URLs directly into the Volume."""
    import urllib.request

    models_dir = f"{VOL_MOUNT}/models"
    os.makedirs(models_dir, exist_ok=True)

    url_map = [
        (dit_url, "anima-preview3-base.safetensors"),
        (qwen3_url, "qwen_3_06b_base.safetensors"),
        (vae_url, "qwen_image_vae.safetensors"),
    ]

    results = []
    for url, fallback_name in url_map:
        if not url:
            results.append(f"Skipped {fallback_name} (no URL)")
            continue
        # Infer filename from URL or use fallback
        url_name = url.split("/")[-1].split("?")[0]
        fname = url_name if url_name.endswith(".safetensors") else fallback_name
        dest = f"{models_dir}/{fname}"

        if os.path.exists(dest):
            results.append(f"Already exists: {fname}")
            continue

        print(f"Downloading {fname} from {url}...")
        try:
            urllib.request.urlretrieve(url, dest)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            results.append(f"Downloaded {fname} ({size_mb:.0f} MB)")
            print(f"  Done: {size_mb:.0f} MB")
        except Exception as e:
            results.append(f"FAILED to download {fname}: {e}")
            continue

    vol.commit()
    return results


@app.function(volumes={VOL_MOUNT: vol}, timeout=1800)
def check_volume_status():
    """Check what's on the Volume — models, datasets, outputs."""
    status = {"models": [], "datasets": [], "outputs": []}

    models_dir = f"{VOL_MOUNT}/models"
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            size_mb = os.path.getsize(f"{models_dir}/{f}") / (1024 * 1024)
            status["models"].append({"name": f, "size_mb": round(size_mb, 1)})

    datasets_dir = f"{VOL_MOUNT}/datasets"
    if os.path.isdir(datasets_dir):
        for d in os.listdir(datasets_dir):
            dpath = f"{datasets_dir}/{d}"
            if os.path.isdir(dpath):
                image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                images = [f for f in os.listdir(dpath) if os.path.splitext(f)[1].lower() in image_exts]
                txts = [f for f in os.listdir(dpath) if f.endswith(".txt")]
                status["datasets"].append({"name": d, "images": len(images), "captions": len(txts)})

    outputs_dir = f"{VOL_MOUNT}/outputs"
    if os.path.isdir(outputs_dir):
        for f in os.listdir(outputs_dir):
            if f.endswith(".safetensors"):
                size_mb = os.path.getsize(f"{outputs_dir}/{f}") / (1024 * 1024)
                status["outputs"].append({"name": f, "size_mb": round(size_mb, 1)})

    return status


# ─── Training Function ────────────────────────────────────────────────────

@app.function(
    image=training_image,
    gpu=DEFAULT_GPU,
    volumes={VOL_MOUNT: vol},
    timeout=14400,  # 4 hours max
)
def train_lora(config: dict):
    """Run Anima LoRA training on a cloud GPU."""
    import subprocess
    import toml as toml_lib

    training_name = config.get("training_name", "my_lora").strip().replace(" ", "_")
    dataset_name = config.get("dataset_name", "")

    # Paths inside the container
    models_dir = f"{VOL_MOUNT}/models"
    dataset_dir = f"{VOL_MOUNT}/datasets/{dataset_name}"
    output_dir = f"{VOL_MOUNT}/outputs"
    work_dir = "/tmp/training"
    sd_scripts = "/opt/sd-scripts"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Verify dataset exists
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Dataset '{dataset_name}' not found on Volume. Upload it first!")

    # Auto-detect DiT model (prefer preview3 > preview2 > preview)
    dit_path = None
    for name in ["anima-preview3-base.safetensors", "anima-preview2.safetensors", "anima-preview.safetensors"]:
        p = f"{models_dir}/{name}"
        if os.path.exists(p):
            dit_path = p
            break
    if not dit_path:
        # Check for any .safetensors in models that starts with "anima"
        for f in os.listdir(models_dir):
            if f.startswith("anima") and f.endswith(".safetensors"):
                dit_path = f"{models_dir}/{f}"
                break
    if not dit_path:
        raise ValueError("No Anima DiT model found in /data/models/. Upload models first!")

    qwen3_path = f"{models_dir}/qwen_3_06b_base.safetensors"
    vae_path = f"{models_dir}/qwen_image_vae.safetensors"

    for p, name in [(qwen3_path, "Qwen3"), (vae_path, "VAE")]:
        if not os.path.exists(p):
            raise ValueError(f"{name} model not found at {p}. Upload models first!")

    # Generate TOML config
    cache_te = config.get("cache_text_encoder_outputs", True)
    shuffle = config.get("shuffle_caption", False) and not cache_te

    toml_data = {
        "general": {
            "shuffle_caption": shuffle,
            "keep_tokens": int(config.get("keep_tokens", 1)),
            "caption_extension": config.get("caption_ext", ".txt"),
        },
    }

    multi_res = config.get("enable_multi_res", False)
    batch_size = int(config.get("train_batch_size", 2))
    num_reps = int(config.get("num_repeats", 6))

    if multi_res:
        # Multi-resolution: 512, 1024, 1536 (like official Anima creator)
        toml_data["datasets"] = []
        for res in [512, 1024, 1536]:
            res_batch = max(1, batch_size * 2) if res == 512 else batch_size if res == 1024 else max(1, batch_size // 2)
            toml_data["datasets"].append({
                "resolution": res,
                "batch_size": res_batch,
                "subsets": [{"image_dir": dataset_dir, "num_repeats": num_reps}],
            })
    else:
        toml_data["datasets"] = [{
            "resolution": int(config.get("resolution", 1024)),
            "batch_size": batch_size,
            "subsets": [{"image_dir": dataset_dir, "num_repeats": num_reps}],
        }]

    toml_path = f"{work_dir}/dataset.toml"
    with open(toml_path, "w") as f:
        toml_lib.dump(toml_data, f)

    # Build training command
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        f"{sd_scripts}/anima_train_network.py",
        f"--pretrained_model_name_or_path={dit_path}",
        f"--qwen3={qwen3_path}",
        f"--vae={vae_path}",
        f"--output_dir={output_dir}",
        f"--output_name={training_name}",
        f"--dataset_config={toml_path}",
        f"--train_batch_size={int(config.get('train_batch_size', 2))}",
        f"--max_train_epochs={int(config.get('max_train_epochs', 12))}",
        f"--resolution={1536 if config.get('enable_multi_res') else int(config.get('resolution', 1024))},{1536 if config.get('enable_multi_res') else int(config.get('resolution', 1024))}",
        f"--optimizer_type={config.get('optimizer', 'AdamW8bit')}",
        f"--learning_rate={float(config.get('learning_rate', 0.0001))}",
        f"--network_dim={int(config.get('network_dim', 8))}",
        f"--network_alpha={int(config.get('network_alpha', 4))}",
        f"--lr_scheduler={config.get('lr_scheduler', 'cosine_with_restarts')}",
        f"--lr_scheduler_num_cycles={int(config.get('lr_scheduler_cycles', 4))}",
        f"--keep_tokens={int(config.get('keep_tokens', 1))}",
        "--save_model_as=safetensors",
        f"--seed={int(config.get('seed', 42))}",
        f"--mixed_precision={config.get('mixed_precision', 'bf16')}",
        "--network_module=networks.lora_anima",
        "--persistent_data_loader_workers",
        f"--timestep_sampling={config.get('timestep_sampling', 'sigmoid')}",
        f"--sigmoid_scale={float(config.get('sigmoid_scale', 1.0))}",
        f"--discrete_flow_shift={float(config.get('discrete_flow_shift', 1.0))}",
        f"--save_every_n_epochs={int(config.get('save_every_n_epochs', 2))}",
    ]

    # Conditional flags
    if config.get("enable_bucket", True):
        cmd.append("--enable_bucket")
    if config.get("bucket_no_upscale", True):
        cmd.append("--bucket_no_upscale")
    if config.get("cache_latents", True):
        cmd.append("--cache_latents")
    if config.get("cache_latents_to_disk", True):
        cmd.append("--cache_latents_to_disk")
    if config.get("cache_text_encoder_outputs", True):
        cmd.extend(["--cache_text_encoder_outputs", "--network_train_unet_only"])
    if config.get("gradient_checkpointing", True):
        cmd.append("--gradient_checkpointing")
    if config.get("xformers", True):
        cmd.append("--xformers")
    if config.get("split_attn", True):
        cmd.append("--split_attn")

    vae_chunk = int(config.get("vae_chunk_size", 64))
    if vae_chunk > 0:
        cmd.append(f"--vae_chunk_size={vae_chunk}")
    if config.get("vae_disable_cache", True):
        cmd.append("--vae_disable_cache")

    blocks = int(config.get("blocks_to_swap", 0))
    if blocks > 0:
        cmd.append(f"--blocks_to_swap={blocks}")

    print(f"\n{'='*60}")
    print(f"  Training: {training_name}")
    print(f"  GPU: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'unknown')}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}\n")
    print("Command:", " ".join(cmd[:5]), "...")

    # Run training with memory fragmentation fix
    import copy
    train_env = copy.deepcopy(os.environ)
    train_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    process = subprocess.run(
        cmd,
        cwd=sd_scripts,
        capture_output=False,
        env=train_env,
    )

    if process.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {process.returncode}")

    # Commit outputs to volume
    vol.commit()

    # List output files
    output_files = []
    if os.path.isdir(output_dir):
        for f in os.listdir(output_dir):
            if training_name in f and f.endswith(".safetensors"):
                size_mb = os.path.getsize(f"{output_dir}/{f}") / (1024 * 1024)
                output_files.append({"name": f, "size_mb": round(size_mb, 1)})

    return {
        "status": "complete",
        "training_name": training_name,
        "output_files": output_files,
    }


# ─── Illustrious 2.0 (SDXL) Training ─────────────────────────────────────

@app.function(
    image=training_image,
    gpu=DEFAULT_GPU,
    volumes={VOL_MOUNT: vol},
    timeout=14400,  # 4 hours max
)
def train_illustrious_lora(config: dict):
    """Run Illustrious 2.0 (SDXL) LoRA training on a cloud GPU."""
    import subprocess
    import toml as toml_lib

    training_name = config.get("training_name", "my_lora").strip().replace(" ", "_")
    dataset_name = config.get("dataset_name", "")

    # Paths inside the container
    models_dir = f"{VOL_MOUNT}/models"
    dataset_dir = f"{VOL_MOUNT}/datasets/{dataset_name}"
    output_dir = f"{VOL_MOUNT}/outputs"
    work_dir = "/tmp/training"
    sd_scripts = "/opt/sd-scripts"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Verify dataset exists
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Dataset '{dataset_name}' not found on Volume. Upload it first!")

    # Auto-detect Illustrious checkpoint
    checkpoint_path = None
    for f in sorted(os.listdir(models_dir)):
        fl = f.lower()
        if fl.endswith(".safetensors") and ("illustrious" in fl or "noob" in fl or "sdxl" in fl):
            checkpoint_path = f"{models_dir}/{f}"
            break
    # Fallback: use explicitly provided model name
    model_name = config.get("illustrious_model", "")
    if model_name and not checkpoint_path:
        p = f"{models_dir}/{model_name}"
        if os.path.exists(p):
            checkpoint_path = p
    if not checkpoint_path:
        raise ValueError(
            "No Illustrious/SDXL checkpoint found in /data/models/. "
            "Upload one first! (filename should contain 'illustrious', 'noob', or 'sdxl')"
        )

    # Generate TOML config
    cache_te = config.get("cache_text_encoder_outputs", True)
    shuffle = config.get("shuffle_caption", False) and not cache_te

    toml_data = {
        "general": {
            "shuffle_caption": shuffle,
            "keep_tokens": int(config.get("keep_tokens", 1)),
            "caption_extension": config.get("caption_ext", ".txt"),
        },
    }

    multi_res = config.get("enable_multi_res", False)
    batch_size = int(config.get("train_batch_size", 4))
    num_reps = int(config.get("num_repeats", 1))

    if multi_res:
        # Multi-resolution: 512, 768, 1024 (SDXL native sizes)
        toml_data["datasets"] = []
        for res in [512, 768, 1024]:
            res_batch = max(1, batch_size * 2) if res == 512 else batch_size if res == 768 else max(1, batch_size // 2)
            toml_data["datasets"].append({
                "resolution": res,
                "batch_size": res_batch,
                "subsets": [{"image_dir": dataset_dir, "num_repeats": num_reps}],
            })
    else:
        toml_data["datasets"] = [{
            "resolution": int(config.get("resolution", 1024)),
            "batch_size": batch_size,
            "subsets": [{"image_dir": dataset_dir, "num_repeats": num_reps}],
        }]

    toml_path = f"{work_dir}/dataset.toml"
    with open(toml_path, "w") as f:
        toml_lib.dump(toml_data, f)

    # Build training command — uses sdxl_train_network.py
    max_train_steps = int(config.get("max_train_steps", 0))

    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        f"{sd_scripts}/sdxl_train_network.py",
        f"--pretrained_model_name_or_path={checkpoint_path}",
        f"--output_dir={output_dir}",
        f"--output_name={training_name}",
        f"--dataset_config={toml_path}",
        f"--train_batch_size={int(config.get('train_batch_size', 4))}",
        f"--resolution={1024 if config.get('enable_multi_res') else int(config.get('resolution', 1024))},{1024 if config.get('enable_multi_res') else int(config.get('resolution', 1024))}",
        f"--optimizer_type={config.get('optimizer', 'AdamW8bit')}",
        f"--learning_rate={float(config.get('learning_rate', 0.0001))}",
        f"--network_dim={int(config.get('network_dim', 32))}",
        f"--network_alpha={int(config.get('network_alpha', 32))}",
        f"--lr_scheduler={config.get('lr_scheduler', 'cosine_with_restarts')}",
        f"--lr_scheduler_num_cycles={int(config.get('lr_scheduler_cycles', 4))}",
        f"--keep_tokens={int(config.get('keep_tokens', 1))}",
        "--save_model_as=safetensors",
        f"--seed={int(config.get('seed', 42))}",
        f"--mixed_precision={config.get('mixed_precision', 'bf16')}",
        "--network_module=networks.lora",
        "--persistent_data_loader_workers",
    ]

    # Use max_train_steps if set, otherwise use epochs
    if max_train_steps > 0:
        cmd.append(f"--max_train_steps={max_train_steps}")
    else:
        cmd.append(f"--max_train_epochs={int(config.get('max_train_epochs', 15))}")

    # Save by steps (not epochs)
    save_every_steps = int(config.get("save_every_n_steps", 250))
    if save_every_steps > 0:
        cmd.append(f"--save_every_n_steps={save_every_steps}")

    max_saves = int(config.get("max_saves", 4))
    if max_saves > 0 and save_every_steps > 0:
        cmd.append(f"--save_last_n_steps={max_saves * save_every_steps}")

    # Conv dimensions + weight decay via --network_args
    conv_dim = int(config.get("conv_dim", 16))
    conv_alpha = int(config.get("conv_alpha", 16))
    weight_decay = float(config.get("weight_decay", 0.0001))
    network_args = [f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}"]
    cmd.extend(["--network_args"] + network_args)

    # Optimizer args (weight decay)
    if weight_decay > 0:
        cmd.extend(["--optimizer_args", f"weight_decay={weight_decay}"])

    # Caption dropout
    caption_dropout = float(config.get("caption_dropout", 0))
    if caption_dropout > 0:
        cmd.append(f"--caption_dropout_rate={caption_dropout}")

    # Conditional flags
    if config.get("enable_bucket", True):
        cmd.append("--enable_bucket")
    if config.get("bucket_no_upscale", True):
        cmd.append("--bucket_no_upscale")
    if config.get("flip_aug", True):
        cmd.append("--flip_aug")
    if config.get("cache_latents", True):
        cmd.append("--cache_latents")
    if config.get("cache_latents_to_disk", True):
        cmd.append("--cache_latents_to_disk")
    if config.get("cache_text_encoder_outputs", False):
        cmd.extend(["--cache_text_encoder_outputs", "--network_train_unet_only"])
    if config.get("gradient_checkpointing", False):
        cmd.append("--gradient_checkpointing")
    if config.get("xformers", True):
        cmd.append("--xformers")

    # Optional SDXL-specific args
    min_snr = float(config.get("min_snr_gamma", 0))
    if min_snr > 0:
        cmd.append(f"--min_snr_gamma={min_snr}")
    noise_offset = float(config.get("noise_offset", 0))
    if noise_offset > 0:
        cmd.append(f"--noise_offset={noise_offset}")

    print(f"\n{'='*60}")
    print(f"  Illustrious Training: {training_name}")
    print(f"  Model: {os.path.basename(checkpoint_path)}")
    print(f"  GPU: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'unknown')}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Steps: {max_train_steps if max_train_steps > 0 else 'auto (epochs)'}")
    print(f"  Dim: {config.get('network_dim', 32)} / Alpha: {config.get('network_alpha', 32)} / Conv: {config.get('conv_dim', 16)}")
    print(f"{'='*60}\n")
    print("Command:", " ".join(cmd[:5]), "...")

    # Run training
    process = subprocess.run(
        cmd,
        cwd=sd_scripts,
        capture_output=False,
    )

    if process.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {process.returncode}")

    # Commit outputs to volume
    vol.commit()

    # List output files
    output_files = []
    if os.path.isdir(output_dir):
        for f in os.listdir(output_dir):
            if training_name in f and f.endswith(".safetensors"):
                size_mb = os.path.getsize(f"{output_dir}/{f}") / (1024 * 1024)
                output_files.append({"name": f, "size_mb": round(size_mb, 1)})

    return {
        "status": "complete",
        "training_name": training_name,
        "model_used": os.path.basename(checkpoint_path),
        "output_files": output_files,
    }


# ─── Download Model to Volume (datacenter-to-datacenter) ────────────────

@app.function(
    image=training_image,
    volumes={VOL_MOUNT: vol},
    timeout=7200,
)
def download_model_to_volume(url: str, dest_subdir: str = "models"):
    """Download a model directly on Modal's servers into the volume.
    Much faster than downloading to local machine then uploading.
    """
    import subprocess

    filename = url.split("/")[-1].split("?")[0]
    if not filename.endswith((".safetensors", ".ckpt", ".pt", ".bin")):
        filename = "downloaded_model.safetensors"

    dest_dir = f"{VOL_MOUNT}/{dest_subdir}"
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = f"{dest_dir}/{filename}"

    print(f"Downloading {url}")
    print(f"  -> {dest_path}")

    result = subprocess.run(
        ["wget", "-O", dest_path, "--progress=bar:force", url],
        capture_output=False, timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"wget download failed for {filename}")

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"Downloaded: {filename} ({size_mb:.1f} MB)")

    vol.commit()
    return {
        "status": "ok",
        "filename": filename,
        "size_mb": round(size_mb, 1),
        "path": dest_path,
    }


# ─── Local entry point for CLI testing ────────────────────────────────────

@app.local_entrypoint()
def main():
    """Quick test: check volume status."""
    print("Checking volume status...")
    status = check_volume_status.remote()
    print(f"Models: {status['models']}")
    print(f"Datasets: {status['datasets']}")
    print(f"Outputs: {status['outputs']}")
