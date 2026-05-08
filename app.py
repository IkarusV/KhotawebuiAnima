"""
Anima LoRA Training Web UI
Flask backend that generates TOML dataset configs and BAT training scripts
for kohya sd-scripts Anima LoRA training.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, send_file

app = Flask(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent.resolve()
HISTORY_DIR = APP_DIR / "history"
OUTPUT_DIR = APP_DIR / "output"
SETTINGS_FILE = APP_DIR / "settings.json"
PRESETS_FILE = APP_DIR / "presets.json"

HISTORY_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODAL_DIR = APP_DIR / "modal"
DOWNLOADS_DIR = APP_DIR / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

# Modal training state
modal_jobs = {}  # job_id -> {status, message, result}

# ─── Default Settings ─────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "sd_scripts_path": r"C:\Aithing\sd-scripts-main",
    "default_dit_model": "",
    "default_qwen3": "",
    "default_vae": "",
}


def load_settings():
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
            merged = {**DEFAULT_SETTINGS, **saved}
            return merged
    return dict(DEFAULT_SETTINGS)


def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


# ─── Default Training Config ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    "training_name": "my_anima_lora",
    "trigger_word": "",
    # Model paths
    "dit_model_path": "",
    "qwen3_path": "",
    "vae_path": "",
    # Dataset
    "dataset_path": "",
    "num_repeats": 10,
    "keep_tokens": 1,
    "shuffle_caption": False,
    "caption_ext": ".txt",
    "resolution": 1024,
    # Training
    "network_dim": 8,
    "network_alpha": 4,
    "learning_rate": 0.0001,
    "optimizer": "AdamW8bit",
    "lr_scheduler": "cosine_with_restarts",
    "lr_scheduler_cycles": 4,
    "max_train_epochs": 20,
    "train_batch_size": 2,
    "timestep_sampling": "sigmoid",
    "sigmoid_scale": 1.0,
    "discrete_flow_shift": 1.0,
    "mixed_precision": "bf16",
    "seed": 42,
    "save_every_n_epochs": 2,
    "max_saves": 5,
    # VRAM Optimization
    "cache_latents": True,
    "cache_latents_to_disk": True,
    "cache_text_encoder_outputs": True,
    "gradient_checkpointing": True,
    "xformers": True,
    "split_attn": True,
    "vae_chunk_size": 64,
    "vae_disable_cache": True,
    "blocks_to_swap": 0,
    # Buckets
    "enable_bucket": True,
    "bucket_no_upscale": True,
}


# ─── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    settings = load_settings()
    return render_template("index.html", defaults=DEFAULT_CONFIG, settings=settings)


@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify(load_settings())


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.json
    settings = load_settings()
    settings.update(data)
    save_settings(settings)
    return jsonify({"status": "ok", "message": "Settings saved!"})


@app.route("/api/generate", methods=["POST"])
def generate_config():
    """Generate .toml and .bat files from the form data."""
    config = request.json

    # Auto-fix: shuffle_caption is incompatible with cache_text_encoder_outputs
    if config.get("cache_text_encoder_outputs") and config.get("shuffle_caption"):
        config["shuffle_caption"] = False

    # Validate required fields
    errors = []
    if not config.get("training_name", "").strip():
        errors.append("Training name is required")
    if not config.get("dataset_path", "").strip():
        errors.append("Dataset path is required")
    if not config.get("dit_model_path", "").strip():
        errors.append("DiT model path is required")
    if not config.get("qwen3_path", "").strip():
        errors.append("Qwen3 text encoder path is required")
    if not config.get("vae_path", "").strip():
        errors.append("VAE path is required")

    if errors:
        return jsonify({"status": "error", "errors": errors}), 400

    training_name = config["training_name"].strip().replace(" ", "_")

    # Generate TOML content
    toml_content = generate_toml(config)

    # Generate BAT content
    settings = load_settings()
    toml_filename = f"{training_name}_dataset.toml"
    toml_path = OUTPUT_DIR / toml_filename
    bat_content = generate_bat(config, settings, str(toml_path))

    # Write files
    toml_path.write_text(toml_content, encoding="utf-8")
    bat_path = OUTPUT_DIR / f"{training_name}_train.bat"
    bat_path.write_text(bat_content, encoding="utf-8")

    # Save to history
    history_id = str(uuid.uuid4())[:8]
    history_entry = {
        "id": history_id,
        "config": config,
        "created_at": datetime.now().isoformat(),
        "toml_file": str(toml_path),
        "bat_file": str(bat_path),
    }
    history_file = HISTORY_DIR / f"{history_id}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_entry, f, indent=2, ensure_ascii=False)

    return jsonify({
        "status": "ok",
        "message": f"Generated files for '{training_name}'",
        "toml_path": str(toml_path),
        "bat_path": str(bat_path),
        "toml_content": toml_content,
        "bat_content": bat_content,
        "history_id": history_id,
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return all saved training configs, newest first."""
    entries = []
    for f in sorted(HISTORY_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                entry = json.load(fh)
                entries.append(entry)
        except Exception:
            continue
    return jsonify(entries)


@app.route("/api/history/<history_id>", methods=["GET"])
def get_history_entry(history_id):
    """Return a single history entry's config."""
    history_file = HISTORY_DIR / f"{history_id}.json"
    if not history_file.exists():
        return jsonify({"status": "error", "message": "Not found"}), 404
    with open(history_file, "r", encoding="utf-8") as f:
        entry = json.load(f)
    return jsonify(entry)


@app.route("/api/history/<history_id>", methods=["DELETE"])
def delete_history_entry(history_id):
    """Delete a history entry."""
    history_file = HISTORY_DIR / f"{history_id}.json"
    if history_file.exists():
        history_file.unlink()
        return jsonify({"status": "ok", "message": "Deleted"})
    return jsonify({"status": "error", "message": "Not found"}), 404


@app.route("/api/scan-dataset", methods=["POST"])
def scan_dataset():
    """Scan a directory and count image files."""
    data = request.json
    folder = data.get("path", "")
    if not folder or not os.path.isdir(folder):
        return jsonify({"status": "error", "count": 0, "message": "Directory not found"})

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = [f for f in os.listdir(folder) if Path(f).suffix.lower() in image_exts]
    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    return jsonify({
        "status": "ok",
        "image_count": len(images),
        "caption_count": len(txt_files),
        "sample_files": sorted(images)[:5],
    })


@app.route("/api/preview", methods=["POST"])
def preview_config():
    """Preview the TOML and BAT content without saving."""
    config = request.json
    settings = load_settings()
    training_name = config.get("training_name", "preview").strip().replace(" ", "_")
    toml_path = str(OUTPUT_DIR / f"{training_name}_dataset.toml")

    toml_content = generate_toml(config)
    bat_content = generate_bat(config, settings, toml_path)

    # Calculate steps
    dataset_path = config.get("dataset_path", "")
    image_count = 0
    if dataset_path and os.path.isdir(dataset_path):
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        image_count = len([f for f in os.listdir(dataset_path) if Path(f).suffix.lower() in image_exts])

    num_repeats = int(config.get("num_repeats", 10))
    epochs = int(config.get("max_train_epochs", 20))
    batch = int(config.get("train_batch_size", 2))
    total_steps = (image_count * num_repeats * epochs) // max(batch, 1) if image_count > 0 else 0

    return jsonify({
        "toml_content": toml_content,
        "bat_content": bat_content,
        "image_count": image_count,
        "total_steps": total_steps,
    })


@app.route("/api/launch", methods=["POST"])
def launch_training():
    """Launch training by running the BAT file."""
    data = request.json
    bat_path = data.get("bat_path", "")
    if not bat_path or not os.path.exists(bat_path):
        return jsonify({"status": "error", "message": "BAT file not found"}), 404

    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "cmd", "/k", bat_path],
            cwd=os.path.dirname(bat_path),
        )
        return jsonify({"status": "ok", "message": "Training launched in new window!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── Modal Cloud API Routes ───────────────────────────────────────────────

def _run_modal_cmd(args, timeout=300):
    """Run a modal CLI command and return (success, output)."""
    venv_python = str(APP_DIR / "venv" / "Scripts" / "python.exe")
    cmd = [venv_python, "-m", "modal"] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(MODAL_DIR)
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


@app.route("/api/modal/status", methods=["GET"])
def modal_status():
    """Check Modal auth and volume status."""
    # Check if modal is installed
    venv_python = str(APP_DIR / "venv" / "Scripts" / "python.exe")
    try:
        result = subprocess.run(
            [venv_python, "-c", "import modal; print(modal.__version__)"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return jsonify({"status": "error", "message": "Modal not installed", "installed": False})
        modal_version = result.stdout.strip()
    except Exception as e:
        return jsonify({"status": "error", "message": str(e), "installed": False})

    # Check auth
    ok, output = _run_modal_cmd(["profile", "current"])
    if not ok:
        return jsonify({"status": "error", "message": "Not authenticated. Run: modal token set", "installed": True, "authenticated": False})

    return jsonify({
        "status": "ok",
        "installed": True,
        "authenticated": True,
        "modal_version": modal_version,
        "profile": output,
    })


@app.route("/api/modal/volume-status", methods=["GET"])
def modal_volume_status():
    """Check what's stored on the Modal Volume using modal volume ls CLI."""
    venv_python = str(APP_DIR / "venv" / "Scripts" / "python.exe")
    status = {"models": [], "datasets": [], "outputs": []}

    def ls_volume(path):
        """List files at a path on the volume, returns list of lines."""
        try:
            result = subprocess.run(
                [venv_python, "-m", "modal", "volume", "ls", "anima-training-data", path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return [], False
            # When captured, output is plain filenames, one per line
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            return lines, True
        except Exception:
            return [], False

    try:
        # Check models/
        lines, ok = ls_volume("models/")
        if ok:
            for line in lines:
                # Lines are like: models/anima-preview2.safetensors
                fname = line.replace("models/", "").strip()
                if fname and fname.endswith(".safetensors"):
                    status["models"].append({"name": fname, "size": ""})

        # Check datasets/
        lines, ok = ls_volume("datasets/")
        if ok:
            for line in lines:
                dname = line.replace("datasets/", "").strip().rstrip("/")
                if dname:
                    # Count files in each dataset
                    ds_lines, ds_ok = ls_volume(f"datasets/{dname}/")
                    img_count = 0
                    txt_count = 0
                    if ds_ok:
                        for dl in ds_lines:
                            dl_lower = dl.lower()
                            if any(ext in dl_lower for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]):
                                img_count += 1
                            if ".txt" in dl_lower:
                                txt_count += 1
                    status["datasets"].append({"name": dname, "images": img_count, "captions": txt_count})

        # Check outputs/
        lines, ok = ls_volume("outputs/")
        if ok:
            for line in lines:
                fname = line.replace("outputs/", "").strip()
                if fname and fname.endswith(".safetensors"):
                    status["outputs"].append({"name": fname, "size": ""})

        return jsonify({"status": "ok", "data": status})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/modal/upload-dataset", methods=["POST"])
def modal_upload_dataset():
    """Upload a local dataset folder to Modal Volume."""
    data = request.json
    local_path = data.get("path", "")
    dataset_name = data.get("name", "").strip().replace(" ", "_")

    if not local_path or not os.path.isdir(local_path):
        return jsonify({"status": "error", "message": "Dataset folder not found"}), 400
    if not dataset_name:
        return jsonify({"status": "error", "message": "Dataset name is required"}), 400

    # Count files
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = os.listdir(local_path)
    images = [f for f in files if Path(f).suffix.lower() in image_exts]
    captions = [f for f in files if f.endswith(".txt")]

    # Upload via modal volume put
    venv_python = str(APP_DIR / "venv" / "Scripts" / "python.exe")
    remote_path = f"datasets/{dataset_name}"

    try:
        # First create the volume if it doesn't exist
        subprocess.run(
            [venv_python, "-m", "modal", "volume", "create", "anima-training-data"],
            capture_output=True, text=True, timeout=30,
        )

        # Upload the entire folder
        result = subprocess.run(
            [venv_python, "-m", "modal", "volume", "put", "anima-training-data",
             local_path, remote_path],
            capture_output=True, text=True, timeout=600,
        )

        if result.returncode != 0:
            return jsonify({"status": "error", "message": f"Upload failed: {result.stderr}"})

        return jsonify({
            "status": "ok",
            "message": f"Uploaded {len(images)} images + {len(captions)} captions as '{dataset_name}'",
            "dataset_name": dataset_name,
            "image_count": len(images),
            "caption_count": len(captions),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/modal/upload-models", methods=["POST"])
def modal_upload_models():
    """Upload base models to Modal Volume from URLs."""
    data = request.json
    dit_url = data.get("dit_url", "")
    qwen3_url = data.get("qwen3_url", "")
    vae_url = data.get("vae_url", "")

    if not any([dit_url, qwen3_url, vae_url]):
        return jsonify({"status": "error", "message": "Provide at least one model URL"}), 400

    # Build a runner script (same pattern as training)
    venv_python = str(APP_DIR / "venv" / "Scripts" / "python.exe")
    script = str(MODAL_DIR / "train_anima_modal.py")

    runner_script = f"""
import sys
sys.path.insert(0, r"{MODAL_DIR}")
from train_anima_modal import app as modal_app, upload_models_from_urls

with modal_app.run():
    result = upload_models_from_urls.remote(
        dit_url="{dit_url}",
        qwen3_url="{qwen3_url}",
        vae_url="{vae_url}"
    )
    print(result)
"""
    runner_path = MODAL_DIR / "_run_upload_models.py"
    runner_path.write_text(runner_script, encoding="utf-8")

    try:
        result = subprocess.run(
            [venv_python, str(runner_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=3600, cwd=str(MODAL_DIR),
        )
        # Clean up runner
        try:
            runner_path.unlink()
        except Exception:
            pass
        return jsonify({
            "status": "ok" if result.returncode == 0 else "error",
            "message": result.stdout + result.stderr,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/modal/train", methods=["POST"])
def modal_train():
    """Launch training on Modal cloud GPU."""
    config = request.json
    training_name = config.get("training_name", "my_lora").strip().replace(" ", "_")
    gpu = config.pop("modal_gpu", "H100")

    # Auto-fix shuffle_caption
    if config.get("cache_text_encoder_outputs") and config.get("shuffle_caption"):
        config["shuffle_caption"] = False

    # Validate
    errors = []
    if not training_name:
        errors.append("Training name is required")
    if not config.get("dataset_name", "").strip():
        errors.append("Dataset name is required (upload dataset first)")
    if errors:
        return jsonify({"status": "error", "errors": errors}), 400

    # Save to history
    history_id = str(uuid.uuid4())[:8]
    history_entry = {
        "id": history_id,
        "config": config,
        "created_at": datetime.now().isoformat(),
        "platform": "modal",
        "gpu": gpu,
    }
    history_file = HISTORY_DIR / f"{history_id}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_entry, f, indent=2, ensure_ascii=False)

    # Build the modal run command
    venv_python = str(APP_DIR / "venv" / "Scripts" / "python.exe")
    script = str(MODAL_DIR / "train_anima_modal.py")

    # Create a temp Python script that calls train_lora with our config
    # Write config to a separate JSON file (avoids backslash escaping issues)
    config_path = MODAL_DIR / f"_config_{training_name}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    runner_script = f"""
import os
import json
import modal

# Set GPU BEFORE importing train_anima_modal (decorator reads this at import time)
os.environ["ANIMA_GPU"] = "{gpu}"

from train_anima_modal import app as modal_app, train_lora

with open("_config_{training_name}.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with modal_app.run():
    result = train_lora.remote(config)
    print("RESULT:" + json.dumps(result))
"""

    runner_path = MODAL_DIR / f"_run_{training_name}.py"
    runner_path.write_text(runner_script, encoding="utf-8")

    # Launch in background
    job_id = history_id
    modal_jobs[job_id] = {"status": "running", "message": "Starting training...", "result": None}

    def run_training():
        try:
            process = subprocess.Popen(
                [venv_python, str(runner_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(MODAL_DIR),
            )
            output_lines = []
            for line in process.stdout:
                output_lines.append(line.strip())
                modal_jobs[job_id]["message"] = line.strip()
                if line.startswith("RESULT:"):
                    try:
                        modal_jobs[job_id]["result"] = json.loads(line[7:])
                    except Exception:
                        pass

            process.wait()
            if process.returncode == 0:
                modal_jobs[job_id]["status"] = "complete"
                modal_jobs[job_id]["message"] = "Training complete!"
            else:
                modal_jobs[job_id]["status"] = "error"
                modal_jobs[job_id]["message"] = "\n".join(output_lines[-10:])
        except Exception as e:
            modal_jobs[job_id]["status"] = "error"
            modal_jobs[job_id]["message"] = str(e)
        finally:
            try:
                runner_path.unlink()
            except Exception:
                pass

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return jsonify({
        "status": "ok",
        "message": f"Cloud training started on {gpu}!",
        "job_id": job_id,
        "history_id": history_id,
    })


@app.route("/api/modal/job/<job_id>", methods=["GET"])
def modal_job_status(job_id):
    """Check the status of a Modal training job."""
    job = modal_jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/modal/outputs", methods=["GET"])
def modal_list_outputs():
    """List trained LoRA files on the Modal Volume."""
    venv_python = str(APP_DIR / "venv" / "Scripts" / "python.exe")
    try:
        result = subprocess.run(
            [venv_python, "-m", "modal", "volume", "ls", "anima-training-data", "outputs/"],
            capture_output=True, text=True, timeout=30,
        )
        files = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line and line.endswith(".safetensors"):
                # Strip the outputs/ prefix so we store just the filename
                fname = line.replace("outputs/", "") if line.startswith("outputs/") else line
                files.append(fname)
        return jsonify({"status": "ok", "files": files})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/modal/download", methods=["POST"])
def modal_download():
    """Download a trained LoRA from Modal Volume and serve via browser."""
    data = request.json
    filename = data.get("filename", "")
    if not filename:
        return jsonify({"status": "error", "message": "Filename required"}), 400

    venv_python = str(APP_DIR / "venv" / "Scripts" / "python.exe")
    # Strip outputs/ prefix if passed (defensive)
    clean_name = filename.replace("outputs/", "") if filename.startswith("outputs/") else filename
    local_path = str(DOWNLOADS_DIR / clean_name)

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [venv_python, "-m", "modal", "volume", "get", "anima-training-data",
             f"outputs/{clean_name}", local_path, "--force"],
            capture_output=True, timeout=600, env=env,
        )
        if result.returncode != 0:
            err_msg = result.stderr.decode("utf-8", errors="replace") if result.stderr else "Unknown error"
            return jsonify({"status": "error", "message": f"Download failed: {err_msg}"})

        return jsonify({
            "status": "ok",
            "message": f"Downloaded {clean_name}",
            "download_url": f"/api/modal/serve/{clean_name}",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/modal/serve/<filename>")
def modal_serve_file(filename):
    """Serve a downloaded LoRA file for browser download."""
    from flask import send_file as flask_send_file
    local_path = DOWNLOADS_DIR / filename
    if not local_path.exists():
        return jsonify({"status": "error", "message": "File not found"}), 404
    return flask_send_file(str(local_path), as_attachment=True, download_name=filename)


# ─── Dataset Tools ────────────────────────────────────────────────────────

@app.route("/api/mirror-dataset", methods=["POST"])
def mirror_dataset():
    """X-axis mirror all images in a dataset folder to double it."""
    from PIL import Image as PILImage
    data = request.json
    folder = data.get("folder", "").strip()

    if not folder or not os.path.isdir(folder):
        return jsonify({"status": "error", "message": f"Folder not found: {folder}"}), 400

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    img_files = sorted([f for f in os.listdir(folder)
                        if Path(f).suffix.lower() in image_exts])

    if not img_files:
        return jsonify({"status": "error", "message": "No images found in folder"}), 400

    original_count = len(img_files)
    created = 0

    for i, img_file in enumerate(img_files):
        src_img = os.path.join(folder, img_file)
        base = Path(img_file).stem
        ext = Path(img_file).suffix

        # New number continues from the last original
        new_num = original_count + i
        new_img_name = f"{new_num:03d}{ext}"
        new_img_path = os.path.join(folder, new_img_name)

        # Skip if already exists (re-run safety)
        if os.path.exists(new_img_path):
            continue

        # Mirror image horizontally
        img = PILImage.open(src_img)
        mirrored = img.transpose(PILImage.FLIP_LEFT_RIGHT)
        mirrored.save(new_img_path)

        # Copy matching text file if exists
        src_txt = os.path.join(folder, base + ".txt")
        new_txt = os.path.join(folder, f"{new_num:03d}.txt")
        if os.path.exists(src_txt) and not os.path.exists(new_txt):
            import shutil
            shutil.copy2(src_txt, new_txt)

        created += 1

    final_count = len([f for f in os.listdir(folder)
                       if Path(f).suffix.lower() in image_exts])

    return jsonify({
        "status": "ok",
        "message": f"Mirrored {created} images. Dataset: {original_count} → {final_count}",
        "original": original_count,
        "final": final_count,
    })


@app.route("/api/modal/set-token", methods=["POST"])
def modal_set_token():
    """Set Modal authentication token (to switch accounts)."""
    data = request.json
    token_id = data.get("token_id", "").strip()
    token_secret = data.get("token_secret", "").strip()

    if not token_id or not token_secret:
        return jsonify({"status": "error", "message": "Both token_id and token_secret are required"}), 400

    venv_python = str(APP_DIR / "venv" / "Scripts" / "python.exe")
    try:
        result = subprocess.run(
            [venv_python, "-m", "modal", "token", "set",
             "--token-id", token_id, "--token-secret", token_secret],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return jsonify({"status": "ok", "message": "Token set successfully!"})
        else:
            return jsonify({"status": "error", "message": f"Failed: {result.stderr or result.stdout}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ─── Upscaler Tool ────────────────────────────────────────────────────────

UPSCALED_DIR = APP_DIR / "upscaled_datasets"
UPSCALED_DIR.mkdir(exist_ok=True)

# Upscaler job state (like modal_jobs)
upscale_jobs = {}


def _get_target_res(w, h):
    """Determine target resolution based on aspect ratio."""
    ratio = w / h
    if 0.9 <= ratio <= 1.1:
        return 1024, 1024, "square"
    elif ratio < 0.9:
        return 832, 1216, "portrait"
    else:
        return 1216, 832, "landscape"


@app.route("/api/upscale/scan", methods=["POST"])
def upscale_scan():
    """Scan a dataset folder and report image resolutions."""
    from PIL import Image as PILImage
    data = request.json
    folder = data.get("path", "").strip()

    if not folder or not os.path.isdir(folder):
        return jsonify({"status": "error", "message": "Folder not found"}), 400

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted([f for f in os.listdir(folder) if Path(f).suffix.lower() in image_exts])
    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    if not files:
        return jsonify({"status": "error", "message": "No images found in folder"}), 400

    images_info = []
    needs_upscale = 0
    already_good = 0

    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            with PILImage.open(fpath) as img:
                w, h = img.size
        except Exception:
            continue

        target_w, target_h, ratio_type = _get_target_res(w, h)
        below = w < target_w or h < target_h

        images_info.append({
            "filename": fname,
            "width": w,
            "height": h,
            "ratio_type": ratio_type,
            "target": f"{target_w}x{target_h}",
            "needs_upscale": below,
        })

        if below:
            needs_upscale += 1
        else:
            already_good += 1

    return jsonify({
        "status": "ok",
        "total_images": len(images_info),
        "total_captions": len(txt_files),
        "needs_upscale": needs_upscale,
        "already_good": already_good,
        "images": images_info,
    })


@app.route("/api/upscale/run", methods=["POST"])
def upscale_run():
    """Full upscale pipeline: upload → Modal Real-ESRGAN → download."""
    data = request.json
    folder = data.get("path", "").strip()
    dataset_name = data.get("name", "").strip().replace(" ", "_")
    # Sanitize: extract just the folder name if user entered a full path
    dataset_name = Path(dataset_name).name if dataset_name else ""
    model_type = data.get("model_type", "anime")  # "anime" or "general"
    upscale_all = data.get("upscale_all", False)  # If true, upscale everything

    if not folder or not os.path.isdir(folder):
        return jsonify({"status": "error", "message": "Folder not found"}), 400
    if not dataset_name:
        return jsonify({"status": "error", "message": "Dataset name is required"}), 400

    job_id = str(uuid.uuid4())[:8]
    upscale_jobs[job_id] = {"status": "running", "message": "Starting upscale...", "result": None}

    def run_upscale():
        try:
            from PIL import Image as PILImage

            image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            all_files = sorted(os.listdir(folder))
            img_files = [f for f in all_files if Path(f).suffix.lower() in image_exts]
            txt_files = [f for f in all_files if f.endswith(".txt")]

            # Determine which images need upscaling
            to_upscale = []
            already_good = []
            for fname in img_files:
                fpath = os.path.join(folder, fname)
                try:
                    with PILImage.open(fpath) as img:
                        w, h = img.size
                    target_w, target_h, _ = _get_target_res(w, h)
                    if upscale_all or w < target_w or h < target_h:
                        to_upscale.append(fname)
                    else:
                        already_good.append(fname)
                except Exception:
                    continue

            if not to_upscale:
                upscale_jobs[job_id]["status"] = "complete"
                upscale_jobs[job_id]["message"] = "No images need upscaling — all already at target resolution!"
                upscale_jobs[job_id]["result"] = {"upscaled": 0, "copied": len(already_good)}
                return

            upscale_jobs[job_id]["message"] = f"Uploading {len(to_upscale)} images to Modal..."

            venv_python = str(APP_DIR / "venv" / "Scripts" / "python.exe")

            # Force UTF-8 encoding for Modal subprocess (prevents charmap errors with ✓ characters on Windows)
            utf8_env = os.environ.copy()
            utf8_env["PYTHONIOENCODING"] = "utf-8"

            # 1. Upload images that need upscaling to Modal volume
            # Create a temp folder with just the images to upscale
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                for fname in to_upscale:
                    src = os.path.join(folder, fname)
                    dst = os.path.join(tmp_dir, fname)
                    shutil.copy2(src, dst)
                # Also copy txt files
                for fname in txt_files:
                    src = os.path.join(folder, fname)
                    dst = os.path.join(tmp_dir, fname)
                    shutil.copy2(src, dst)

                # Upload to Modal volume
                result = subprocess.run(
                    [venv_python, "-m", "modal", "volume", "put", "--force", "anima-training-data",
                     tmp_dir, f"upscale_input/{dataset_name}"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
                    env=utf8_env,
                )
                if result.returncode != 0:
                    upscale_jobs[job_id]["status"] = "error"
                    upscale_jobs[job_id]["message"] = f"Upload failed: {result.stderr}"
                    return

            upscale_jobs[job_id]["message"] = f"Running Real-ESRGAN on {len(to_upscale)} images..."

            # 2. Run the upscaler on Modal
            script = str(MODAL_DIR / "upscale_modal.py")
            runner_script = f"""
import json
import modal
from upscale_modal import app as upscale_app, upscale_images

with upscale_app.run():
    result = upscale_images.remote("{dataset_name}", "{model_type}")
    print("RESULT:" + json.dumps(result))
"""
            runner_path = MODAL_DIR / f"_run_upscale_{dataset_name}.py"
            runner_path.write_text(runner_script, encoding="utf-8")

            process = subprocess.Popen(
                [venv_python, str(runner_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", cwd=str(MODAL_DIR),
                env=utf8_env,
            )

            modal_result = None
            for line in process.stdout:
                line = line.strip()
                if line.startswith("RESULT:"):
                    try:
                        modal_result = json.loads(line[7:])
                    except Exception:
                        pass
                elif line:
                    upscale_jobs[job_id]["message"] = line

            process.wait()

            try:
                runner_path.unlink()
            except Exception:
                pass

            if process.returncode != 0 or not modal_result:
                upscale_jobs[job_id]["status"] = "error"
                upscale_jobs[job_id]["message"] = "Modal upscale failed"
                return

            # 3. Download results from Modal volume
            upscale_jobs[job_id]["message"] = "Downloading upscaled images..."
            out_dir = str(UPSCALED_DIR / dataset_name)
            os.makedirs(out_dir, exist_ok=True)

            result = subprocess.run(
                [venv_python, "-m", "modal", "volume", "get", "anima-training-data",
                 f"upscale_output/{dataset_name}", out_dir, "--force"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
                env=utf8_env,
            )

            if result.returncode != 0:
                upscale_jobs[job_id]["status"] = "error"
                upscale_jobs[job_id]["message"] = f"Download failed: {result.stderr}"
                return

            # Flatten nested subfolder from modal volume get (it creates dataset_name/dataset_name/)
            nested_dir = os.path.join(out_dir, dataset_name)
            if os.path.isdir(nested_dir):
                for f in os.listdir(nested_dir):
                    src = os.path.join(nested_dir, f)
                    dst = os.path.join(out_dir, f)
                    if os.path.isfile(src):
                        shutil.move(src, dst)
                try:
                    os.rmdir(nested_dir)
                except Exception:
                    pass

            # 4. Copy already-good images + their captions (skip if already exists from upscale)
            for fname in already_good:
                src = os.path.join(folder, fname)
                dst = os.path.join(out_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

            # 5. Copy ALL caption txt files from source (single source of truth)
            for fname in txt_files:
                src = os.path.join(folder, fname)
                dst = os.path.join(out_dir, fname)
                shutil.copy2(src, dst)  # Always overwrite to ensure source is truth

            upscale_jobs[job_id]["status"] = "complete"
            upscale_jobs[job_id]["message"] = (
                f"Done! {len(to_upscale)} images upscaled, {len(already_good)} copied as-is. "
                f"Output: upscaled_datasets/{dataset_name}/"
            )
            upscale_jobs[job_id]["result"] = {
                "upscaled": len(to_upscale),
                "copied": len(already_good),
                "output_dir": out_dir,
                "modal_result": modal_result,
            }

        except Exception as e:
            upscale_jobs[job_id]["status"] = "error"
            upscale_jobs[job_id]["message"] = str(e)

    thread = threading.Thread(target=run_upscale, daemon=True)
    thread.start()

    return jsonify({
        "status": "ok",
        "message": "Upscale job started!",
        "job_id": job_id,
    })


@app.route("/api/upscale/job/<job_id>", methods=["GET"])
def upscale_job_status(job_id):
    """Check upscale job progress."""
    job = upscale_jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    return jsonify(job)


# ─── Generators ───────────────────────────────────────────────────────────

def generate_toml(config):
    """Generate the dataset TOML config file content."""
    # Auto-fix: shuffle_caption is incompatible with cache_text_encoder_outputs
    cache_te = config.get("cache_text_encoder_outputs", True)
    shuffle = config.get("shuffle_caption", False) and not cache_te
    shuffle_str = "true" if shuffle else "false"

    keep_tokens = int(config.get("keep_tokens", 1))
    caption_ext = config.get("caption_ext", ".txt")
    resolution = int(config.get("resolution", 1024))
    batch_size = int(config.get("train_batch_size", 2))
    num_repeats = int(config.get("num_repeats", 10))
    dataset_path = config.get("dataset_path", "").replace("\\", "/")
    multi_res = config.get("enable_multi_res", False)

    lines = [
        "[general]",
        f"shuffle_caption = {shuffle_str}",
        f"keep_tokens = {keep_tokens}",
        f'caption_extension = "{caption_ext}"',
    ]

    if multi_res:
        # Multi-resolution: 3 dataset blocks at 512, 1024, 1536 (like official Anima creator)
        for res in [512, 1024, 1536]:
            # Lower batch for higher res to manage VRAM
            res_batch = max(1, batch_size * 2) if res == 512 else batch_size if res == 1024 else max(1, batch_size // 2)
            lines.extend([
                "",
                "[[datasets]]",
                f"resolution = {res}",
                f"batch_size = {res_batch}",
                "",
                "  [[datasets.subsets]]",
                f'  image_dir = "{dataset_path}"',
                f"  num_repeats = {num_repeats}",
            ])
    else:
        lines.extend([
            "",
            "[[datasets]]",
            f"resolution = {resolution}",
            f"batch_size = {batch_size}",
            "",
            "  [[datasets.subsets]]",
            f'  image_dir = "{dataset_path}"',
            f"  num_repeats = {num_repeats}",
        ])
    return "\n".join(lines) + "\n"


def generate_bat(config, settings, toml_path):
    """Generate the training BAT script content."""
    sd_scripts_path = settings.get("sd_scripts_path", r"C:\Aithing\sd-scripts")
    venv_activate = os.path.join(sd_scripts_path, "venv", "Scripts", "activate")
    train_script = os.path.join(sd_scripts_path, "anima_train_network.py")

    training_name = config.get("training_name", "my_lora").strip().replace(" ", "_")
    output_dir = str(OUTPUT_DIR)

    # Model paths
    dit = config.get("dit_model_path", "")
    qwen3 = config.get("qwen3_path", "")
    vae = config.get("vae_path", "")

    # Training params
    dim = int(config.get("network_dim", 8))
    alpha = int(config.get("network_alpha", 4))
    lr = float(config.get("learning_rate", 0.0001))
    optimizer = config.get("optimizer", "AdamW8bit")
    scheduler = config.get("lr_scheduler", "cosine_with_restarts")
    scheduler_cycles = int(config.get("lr_scheduler_cycles", 4))
    epochs = int(config.get("max_train_epochs", 20))
    batch = int(config.get("train_batch_size", 2))
    resolution = int(config.get("resolution", 1024))
    timestep = config.get("timestep_sampling", "sigmoid")
    sigmoid_scale = float(config.get("sigmoid_scale", 1.0))
    flow_shift = float(config.get("discrete_flow_shift", 1.0))
    precision = config.get("mixed_precision", "bf16")
    seed = int(config.get("seed", 42))
    save_every = int(config.get("save_every_n_epochs", 2))

    # Build command parts
    parts = [
        "@echo off",
        f'call "{venv_activate}"',
        "",
        f'accelerate launch --num_cpu_threads_per_process 1 "{train_script}" ^',
        f'--pretrained_model_name_or_path="{dit}" ^',
        f'--qwen3="{qwen3}" ^',
        f'--vae="{vae}" ^',
        f'--output_dir="{output_dir}" ^',
        f'--output_name="{training_name}" ^',
        f'--dataset_config="{toml_path}" ^',
        f"--train_batch_size={batch} ^",
        f"--max_train_epochs={epochs} ^",
        f"--resolution={1536 if config.get('enable_multi_res') else resolution},{1536 if config.get('enable_multi_res') else resolution} ^",
        f'--optimizer_type={optimizer} ^',
        f"--learning_rate={lr} ^",
        f"--network_dim={dim} ^",
        f"--network_alpha={alpha} ^",
        f"--lr_scheduler={scheduler} ^",
        f"--lr_scheduler_num_cycles={scheduler_cycles} ^",
        f"--keep_tokens={int(config.get('keep_tokens', 1))} ^",
        "--save_model_as=safetensors ^",
        f"--seed={seed} ^",
        f"--mixed_precision={precision} ^",
        "--network_module=networks.lora_anima ^",
        "--persistent_data_loader_workers ^",
        f'--timestep_sampling="{timestep}" ^',
        f"--sigmoid_scale={sigmoid_scale} ^",
        f"--discrete_flow_shift={flow_shift} ^",
        f"--save_every_n_epochs={save_every} ^",
    ]

    # Conditional flags
    if config.get("enable_bucket", True):
        parts.append("--enable_bucket ^")
    if config.get("bucket_no_upscale", True):
        parts.append("--bucket_no_upscale ^")
    if config.get("cache_latents", True):
        parts.append("--cache_latents ^")
    if config.get("cache_latents_to_disk", True):
        parts.append("--cache_latents_to_disk ^")
    if config.get("cache_text_encoder_outputs", True):
        parts.append("--cache_text_encoder_outputs ^")
        parts.append("--network_train_unet_only ^")
    if config.get("gradient_checkpointing", True):
        parts.append("--gradient_checkpointing ^")
    if config.get("xformers", True):
        parts.append("--xformers ^")
    if config.get("split_attn", True):
        parts.append("--split_attn ^")

    vae_chunk = int(config.get("vae_chunk_size", 64))
    if vae_chunk > 0:
        parts.append(f"--vae_chunk_size={vae_chunk} ^")
    if config.get("vae_disable_cache", True):
        parts.append("--vae_disable_cache ^")

    blocks = int(config.get("blocks_to_swap", 0))
    if blocks > 0:
        parts.append(f"--blocks_to_swap={blocks} ^")

    # Remove trailing ^ from last line
    parts[-1] = parts[-1].rstrip(" ^")

    parts.append("")
    parts.append("echo.")
    parts.append("echo Training complete!")
    parts.append("pause")

    return "\n".join(parts) + "\n"


# ─── Illustrious 2.0 (SDXL) Routes ───────────────────────────────────────

ILLUSTRIOUS_DEFAULTS = {
    "training_name": "my_illustrious_lora",
    "trigger_word": "",
    "checkpoint_path": "",
    "dataset_path": "",
    "num_repeats": 1,
    "keep_tokens": 1,
    "shuffle_caption": False,
    "caption_ext": ".txt",
    "resolution": 1024,
    "network_dim": 16,
    "network_alpha": 16,
    "conv_dim": 16,
    "conv_alpha": 16,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "optimizer": "AdamW8bit",
    "lr_scheduler": "cosine_with_restarts",
    "lr_scheduler_cycles": 4,
    "max_train_steps": 600,
    "max_train_epochs": 15,
    "train_batch_size": 6,
    "mixed_precision": "bf16",
    "seed": 42,
    "save_every_n_steps": 150,
    "max_saves": 4,
    "enable_bucket": True,
    "bucket_no_upscale": True,
    "flip_aug": False,
    "cache_latents": True,
    "cache_latents_to_disk": True,
    "cache_text_encoder_outputs": False,
    "gradient_checkpointing": False,
    "xformers": True,
    "min_snr_gamma": 0,
    "noise_offset": 0,
    "caption_dropout": 0.05,
}


@app.route("/api/illustrious/generate", methods=["POST"])
def illustrious_generate():
    """Generate Illustrious .toml and .bat files."""
    config = request.json

    if config.get("cache_text_encoder_outputs") and config.get("shuffle_caption"):
        config["shuffle_caption"] = False

    errors = []
    if not config.get("training_name", "").strip():
        errors.append("Training name is required")
    if not config.get("dataset_path", "").strip():
        errors.append("Dataset path is required")
    if not config.get("checkpoint_path", "").strip():
        errors.append("Illustrious checkpoint path is required")
    if errors:
        return jsonify({"status": "error", "errors": errors}), 400

    training_name = config["training_name"].strip().replace(" ", "_")

    toml_content = generate_illustrious_toml(config)
    settings = load_settings()
    toml_filename = f"{training_name}_illustrious_dataset.toml"
    toml_path = OUTPUT_DIR / toml_filename
    bat_content = generate_illustrious_bat(config, settings, str(toml_path))

    toml_path.write_text(toml_content, encoding="utf-8")
    bat_path = OUTPUT_DIR / f"{training_name}_illustrious_train.bat"
    bat_path.write_text(bat_content, encoding="utf-8")

    history_id = str(uuid.uuid4())[:8]
    history_entry = {
        "id": history_id,
        "config": config,
        "created_at": datetime.now().isoformat(),
        "toml_file": str(toml_path),
        "bat_file": str(bat_path),
        "platform": "local",
        "model_type": "illustrious",
    }
    history_file = HISTORY_DIR / f"{history_id}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_entry, f, indent=2, ensure_ascii=False)

    return jsonify({
        "status": "ok",
        "message": f"Generated Illustrious files for '{training_name}'",
        "toml_path": str(toml_path),
        "bat_path": str(bat_path),
        "toml_content": toml_content,
        "bat_content": bat_content,
        "history_id": history_id,
    })


@app.route("/api/illustrious/preview", methods=["POST"])
def illustrious_preview():
    """Preview Illustrious TOML and BAT."""
    config = request.json
    settings = load_settings()
    training_name = config.get("training_name", "preview").strip().replace(" ", "_")
    toml_path = str(OUTPUT_DIR / f"{training_name}_illustrious_dataset.toml")

    toml_content = generate_illustrious_toml(config)
    bat_content = generate_illustrious_bat(config, settings, toml_path)

    dataset_path = config.get("dataset_path", "")
    image_count = 0
    if dataset_path and os.path.isdir(dataset_path):
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        image_count = len([f for f in os.listdir(dataset_path)
                          if Path(f).suffix.lower() in image_exts])

    num_repeats = int(config.get("num_repeats", 2))
    epochs = int(config.get("max_train_epochs", 15))
    batch = int(config.get("train_batch_size", 4))
    total_steps = (image_count * num_repeats * epochs) // max(batch, 1) if image_count > 0 else 0

    return jsonify({
        "toml_content": toml_content,
        "bat_content": bat_content,
        "image_count": image_count,
        "total_steps": total_steps,
    })


@app.route("/api/modal/train-illustrious", methods=["POST"])
def modal_train_illustrious():
    """Launch Illustrious training on Modal cloud GPU."""
    config = request.json
    training_name = config.get("training_name", "my_lora").strip().replace(" ", "_")
    gpu = config.pop("modal_gpu", "H100")

    if config.get("cache_text_encoder_outputs") and config.get("shuffle_caption"):
        config["shuffle_caption"] = False

    errors = []
    if not training_name:
        errors.append("Training name is required")
    if not config.get("dataset_name", "").strip():
        errors.append("Dataset name is required (upload dataset first)")
    if errors:
        return jsonify({"status": "error", "errors": errors}), 400

    history_id = str(uuid.uuid4())[:8]
    history_entry = {
        "id": history_id,
        "config": config,
        "created_at": datetime.now().isoformat(),
        "platform": "modal",
        "model_type": "illustrious",
        "gpu": gpu,
    }
    history_file = HISTORY_DIR / f"{history_id}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_entry, f, indent=2, ensure_ascii=False)

    venv_python = str(APP_DIR / "venv" / "Scripts" / "python.exe")

    config_path = MODAL_DIR / f"_config_ill_{training_name}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    runner_script = f"""
import os
import json
import modal

os.environ["ANIMA_GPU"] = "{gpu}"

from train_anima_modal import app as modal_app, train_illustrious_lora

with open("_config_ill_{training_name}.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with modal_app.run():
    result = train_illustrious_lora.remote(config)
    print("RESULT:" + json.dumps(result))
"""

    runner_path = MODAL_DIR / f"_run_ill_{training_name}.py"
    runner_path.write_text(runner_script, encoding="utf-8")

    job_id = history_id
    modal_jobs[job_id] = {"status": "running", "message": "Starting Illustrious training...", "result": None}

    def run_training():
        try:
            process = subprocess.Popen(
                [venv_python, str(runner_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(MODAL_DIR),
            )
            output_lines = []
            for line in process.stdout:
                output_lines.append(line.strip())
                modal_jobs[job_id]["message"] = line.strip()
                if line.startswith("RESULT:"):
                    try:
                        modal_jobs[job_id]["result"] = json.loads(line[7:])
                    except Exception:
                        pass

            process.wait()
            if process.returncode == 0:
                modal_jobs[job_id]["status"] = "complete"
                modal_jobs[job_id]["message"] = "Illustrious training complete!"
            else:
                modal_jobs[job_id]["status"] = "error"
                modal_jobs[job_id]["message"] = "\n".join(output_lines[-10:])
        except Exception as e:
            modal_jobs[job_id]["status"] = "error"
            modal_jobs[job_id]["message"] = str(e)
        finally:
            try:
                runner_path.unlink()
            except Exception:
                pass

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return jsonify({
        "status": "ok",
        "message": f"Illustrious cloud training started on {gpu}!",
        "job_id": job_id,
        "history_id": history_id,
    })


@app.route("/api/modal/upload-illustrious-model", methods=["POST"])
def modal_upload_illustrious_model():
    """Upload an Illustrious/SDXL checkpoint to Modal Volume.
    Downloads directly on Modal's servers (datacenter-to-datacenter).
    """
    data = request.json
    model_url = data.get("model_url", "").strip()

    if not model_url:
        return jsonify({"status": "error", "message": "Model URL is required"}), 400

    venv_python = str(APP_DIR / "venv" / "Scripts" / "python.exe")

    # Create a runner script that downloads directly on Modal
    filename = model_url.split("/")[-1].split("?")[0]
    if not filename.endswith(".safetensors"):
        filename = "illustrious_model.safetensors"

    try:
        job_id = f"ill_upload_{uuid.uuid4().hex[:6]}"
        modal_jobs[job_id] = {"status": "running", "message": f"Downloading {filename} on Modal cloud...", "result": None}

        runner_script = f"""
import json
import modal
from train_anima_modal import app as modal_app, download_model_to_volume

with modal_app.run():
    result = download_model_to_volume.remote("{model_url}", "models")
    print("RESULT:" + json.dumps(result))
"""
        runner_path = MODAL_DIR / f"_run_download_{job_id}.py"
        runner_path.write_text(runner_script, encoding="utf-8")

        def run_download():
            try:
                process = subprocess.Popen(
                    [venv_python, str(runner_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(MODAL_DIR),
                )
                output_lines = []
                for line in process.stdout:
                    output_lines.append(line.strip())
                    modal_jobs[job_id]["message"] = line.strip()
                    if line.startswith("RESULT:"):
                        try:
                            modal_jobs[job_id]["result"] = json.loads(line[7:])
                        except Exception:
                            pass

                process.wait()
                if process.returncode == 0 and modal_jobs[job_id].get("result"):
                    r = modal_jobs[job_id]["result"]
                    modal_jobs[job_id]["status"] = "complete"
                    modal_jobs[job_id]["message"] = f"Uploaded {r.get('filename', filename)} ({r.get('size_mb', '?')} MB) to Modal volume"
                elif process.returncode == 0:
                    modal_jobs[job_id]["status"] = "complete"
                    modal_jobs[job_id]["message"] = f"Uploaded {filename} to Modal volume"
                else:
                    modal_jobs[job_id]["status"] = "error"
                    modal_jobs[job_id]["message"] = "\n".join(output_lines[-10:])
            except Exception as e:
                modal_jobs[job_id]["status"] = "error"
                modal_jobs[job_id]["message"] = str(e)
            finally:
                try:
                    runner_path.unlink()
                except Exception:
                    pass

        thread = threading.Thread(target=run_download, daemon=True)
        thread.start()

        return jsonify({
            "status": "ok",
            "message": f"Downloading {filename} directly on Modal cloud (datacenter-to-datacenter)...",
            "job_id": job_id,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ─── Illustrious Generators ──────────────────────────────────────────────

def generate_illustrious_toml(config):
    """Generate dataset TOML for Illustrious/SDXL training."""
    cache_te = config.get("cache_text_encoder_outputs", True)
    shuffle = config.get("shuffle_caption", False) and not cache_te
    shuffle_str = "true" if shuffle else "false"

    keep_tokens = int(config.get("keep_tokens", 1))
    caption_ext = config.get("caption_ext", ".txt")
    resolution = int(config.get("resolution", 1024))
    batch_size = int(config.get("train_batch_size", 4))
    num_repeats = int(config.get("num_repeats", 2))
    dataset_path = config.get("dataset_path", "").replace("\\", "/")
    multi_res = config.get("enable_multi_res", False)

    lines = [
        "[general]",
        f"shuffle_caption = {shuffle_str}",
        f"keep_tokens = {keep_tokens}",
        f'caption_extension = "{caption_ext}"',
    ]

    if multi_res:
        # Multi-resolution: 3 dataset blocks at 512, 768, 1024 (SDXL native sizes)
        for res in [512, 768, 1024]:
            res_batch = max(1, batch_size * 2) if res == 512 else batch_size if res == 768 else max(1, batch_size // 2)
            lines.extend([
                "",
                "[[datasets]]",
                f"resolution = {res}",
                f"batch_size = {res_batch}",
                "",
                "  [[datasets.subsets]]",
                f'  image_dir = "{dataset_path}"',
                f"  num_repeats = {num_repeats}",
            ])
    else:
        lines.extend([
            "",
            "[[datasets]]",
            f"resolution = {resolution}",
            f"batch_size = {batch_size}",
            "",
            "  [[datasets.subsets]]",
            f'  image_dir = "{dataset_path}"',
            f"  num_repeats = {num_repeats}",
        ])
    return "\n".join(lines) + "\n"


def generate_illustrious_bat(config, settings, toml_path):
    """Generate training BAT script for Illustrious/SDXL."""
    sd_scripts_path = settings.get("sd_scripts_path", r"C:\Aithing\sd-scripts")
    venv_activate = os.path.join(sd_scripts_path, "venv", "Scripts", "activate")
    train_script = os.path.join(sd_scripts_path, "sdxl_train_network.py")

    training_name = config.get("training_name", "my_lora").strip().replace(" ", "_")
    output_dir = str(OUTPUT_DIR)
    checkpoint = config.get("checkpoint_path", "")

    dim = int(config.get("network_dim", 32))
    alpha = int(config.get("network_alpha", 32))
    conv_dim = int(config.get("conv_dim", 16))
    conv_alpha = int(config.get("conv_alpha", 16))
    lr = float(config.get("learning_rate", 0.0001))
    weight_decay = float(config.get("weight_decay", 0.0001))
    optimizer = config.get("optimizer", "AdamW8bit")
    scheduler = config.get("lr_scheduler", "cosine_with_restarts")
    scheduler_cycles = int(config.get("lr_scheduler_cycles", 4))
    max_train_steps = int(config.get("max_train_steps", 0))
    epochs = int(config.get("max_train_epochs", 15))
    batch = int(config.get("train_batch_size", 4))
    resolution = int(config.get("resolution", 1024))
    precision = config.get("mixed_precision", "bf16")
    seed = int(config.get("seed", 42))
    save_every_steps = int(config.get("save_every_n_steps", 250))
    max_saves = int(config.get("max_saves", 4))
    caption_dropout = float(config.get("caption_dropout", 0))

    parts = [
        "@echo off",
        f'call "{venv_activate}"',
        "",
        f'accelerate launch --num_cpu_threads_per_process 1 "{train_script}" ^',
        f'--pretrained_model_name_or_path="{checkpoint}" ^',
        f'--output_dir="{output_dir}" ^',
        f'--output_name="{training_name}" ^',
        f'--dataset_config="{toml_path}" ^',
        f"--train_batch_size={batch} ^",
    ]

    # Use max_train_steps if set, otherwise use epochs
    if max_train_steps > 0:
        parts.append(f"--max_train_steps={max_train_steps} ^")
    else:
        parts.append(f"--max_train_epochs={epochs} ^")

    parts.extend([
        f"--resolution={1024 if config.get('enable_multi_res') else resolution},{1024 if config.get('enable_multi_res') else resolution} ^",
        f"--optimizer_type={optimizer} ^",
        f"--learning_rate={lr} ^",
        f"--network_dim={dim} ^",
        f"--network_alpha={alpha} ^",
        f"--lr_scheduler={scheduler} ^",
        f"--lr_scheduler_num_cycles={scheduler_cycles} ^",
        f"--keep_tokens={int(config.get('keep_tokens', 1))} ^",
        "--save_model_as=safetensors ^",
        f"--seed={seed} ^",
        f"--mixed_precision={precision} ^",
        "--network_module=networks.lora ^",
        "--persistent_data_loader_workers ^",
    ])

    # Save by steps (not epochs)
    if save_every_steps > 0:
        parts.append(f"--save_every_n_steps={save_every_steps} ^")

    if max_saves > 0:
        parts.append(f"--save_last_n_steps={max_saves * save_every_steps} ^")

    # Conv dimensions via --network_args (Kohya format)
    parts.append(f"--network_args conv_dim={conv_dim} conv_alpha={conv_alpha} ^")

    # Optimizer args (weight decay)
    if weight_decay > 0:
        parts.append(f"--optimizer_args weight_decay={weight_decay} ^")

    # Caption dropout
    if caption_dropout > 0:
        parts.append(f"--caption_dropout_rate={caption_dropout} ^")

    if config.get("enable_bucket", True):
        parts.append("--enable_bucket ^")
    if config.get("bucket_no_upscale", True):
        parts.append("--bucket_no_upscale ^")
    if config.get("flip_aug", True):
        parts.append("--flip_aug ^")
    if config.get("cache_latents", True):
        parts.append("--cache_latents ^")
    if config.get("cache_latents_to_disk", True):
        parts.append("--cache_latents_to_disk ^")
    if config.get("cache_text_encoder_outputs", False):
        parts.append("--cache_text_encoder_outputs ^")
        parts.append("--network_train_unet_only ^")
    if config.get("gradient_checkpointing", False):
        parts.append("--gradient_checkpointing ^")
    if config.get("xformers", True):
        parts.append("--xformers ^")

    min_snr = float(config.get("min_snr_gamma", 0))
    if min_snr > 0:
        parts.append(f"--min_snr_gamma={min_snr} ^")
    noise_offset = float(config.get("noise_offset", 0))
    if noise_offset > 0:
        parts.append(f"--noise_offset={noise_offset} ^")

    # Remove trailing ^
    parts[-1] = parts[-1].rstrip(" ^")

    parts.append("")
    parts.append("echo.")
    parts.append("echo Illustrious training complete!")
    parts.append("pause")

    return "\n".join(parts) + "\n"


# ─── Local Image Resizer ──────────────────────────────────────────────────

def _get_resize_target(w, h):
    """Determine target resolution based on aspect ratio."""
    ratio = w / h
    if 0.9 <= ratio <= 1.1:
        return 1024, 1024, "square"
    elif ratio < 0.9:
        return 832, 1216, "portrait"
    else:
        return 1216, 832, "landscape"


@app.route("/api/resize/scan", methods=["POST"])
def resize_scan():
    """Scan a dataset folder and report which images are oversized."""
    from PIL import Image as PILImage
    data = request.json
    folder = data.get("path", "").strip()

    if not folder or not os.path.isdir(folder):
        return jsonify({"status": "error", "message": "Folder not found"}), 400

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted([f for f in os.listdir(folder) if Path(f).suffix.lower() in image_exts])

    images = []
    needs_resize = 0
    already_good = 0
    too_small = 0

    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            with PILImage.open(fpath) as img:
                w, h = img.size
        except Exception:
            continue

        target_w, target_h, ratio_type = _get_resize_target(w, h)

        # Check if image is larger than target in BOTH dimensions
        is_oversized = (w > target_w or h > target_h)
        is_at_target = (w == target_w and h == target_h)
        is_too_small = (w < target_w and h < target_h)

        if is_oversized:
            status = "oversized"
            needs_resize += 1
        elif is_at_target:
            status = "ok"
            already_good += 1
        elif is_too_small:
            status = "small"
            too_small += 1
        else:
            status = "ok"
            already_good += 1

        images.append({
            "filename": fname,
            "width": w,
            "height": h,
            "ratio_type": ratio_type,
            "target": f"{target_w}×{target_h}",
            "status": status,
        })

    return jsonify({
        "status": "ok",
        "total_images": len(images),
        "needs_resize": needs_resize,
        "already_good": already_good,
        "too_small": too_small,
        "images": images,
    })


@app.route("/api/resize/run", methods=["POST"])
def resize_run():
    """Resize oversized images in-place using Lanczos."""
    from PIL import Image as PILImage
    data = request.json
    folder = data.get("path", "").strip()

    if not folder or not os.path.isdir(folder):
        return jsonify({"status": "error", "message": "Folder not found"}), 400

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted([f for f in os.listdir(folder) if Path(f).suffix.lower() in image_exts])

    resized = 0
    skipped = 0
    errors = 0

    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            with PILImage.open(fpath) as img:
                w, h = img.size
                target_w, target_h, _ = _get_resize_target(w, h)

                # Only resize if oversized
                if w <= target_w and h <= target_h:
                    skipped += 1
                    continue

                # Resize with Lanczos (highest quality for downscaling)
                resized_img = img.resize((target_w, target_h), PILImage.LANCZOS)

                # Save in original format
                ext = Path(fname).suffix.lower()
                if ext in (".jpg", ".jpeg"):
                    resized_img.save(fpath, "JPEG", quality=95)
                elif ext == ".webp":
                    resized_img.save(fpath, "WEBP", quality=95)
                elif ext == ".bmp":
                    resized_img.save(fpath, "BMP")
                else:
                    resized_img.save(fpath, "PNG")

                resized += 1
                print(f"  Resized {fname}: {w}×{h} → {target_w}×{target_h}")
        except Exception as e:
            print(f"  Error resizing {fname}: {e}")
            errors += 1

    return jsonify({
        "status": "ok",
        "resized": resized,
        "skipped": skipped,
        "errors": errors,
    })


# ─── Presets System ───────────────────────────────────────────────────────

def _load_presets():
    """Load presets from disk."""
    if PRESETS_FILE.exists():
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"anima": {}, "illustrious": {}}


def _save_presets(presets):
    """Save presets to disk."""
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)


@app.route("/api/presets", methods=["GET"])
def list_presets():
    """List all saved presets for both tabs."""
    presets = _load_presets()
    return jsonify({
        "status": "ok",
        "anima": list(presets.get("anima", {}).keys()),
        "illustrious": list(presets.get("illustrious", {}).keys()),
    })


@app.route("/api/presets/<tab>/<name>", methods=["GET"])
def get_preset(tab, name):
    """Get a specific preset's config."""
    if tab not in ("anima", "illustrious"):
        return jsonify({"status": "error", "message": "Tab must be 'anima' or 'illustrious'"}), 400

    presets = _load_presets()
    config = presets.get(tab, {}).get(name)
    if not config:
        return jsonify({"status": "error", "message": f"Preset '{name}' not found"}), 404

    return jsonify({"status": "ok", "name": name, "tab": tab, "config": config})


@app.route("/api/presets/<tab>/<name>", methods=["POST"])
def save_preset(tab, name):
    """Save or update a preset."""
    if tab not in ("anima", "illustrious"):
        return jsonify({"status": "error", "message": "Tab must be 'anima' or 'illustrious'"}), 400

    if not name or len(name) > 50:
        return jsonify({"status": "error", "message": "Preset name must be 1-50 characters"}), 400

    config = request.json
    if not config:
        return jsonify({"status": "error", "message": "No config data provided"}), 400

    presets = _load_presets()
    if tab not in presets:
        presets[tab] = {}

    is_new = name not in presets[tab]
    presets[tab][name] = config
    _save_presets(presets)

    action = "created" if is_new else "updated"
    return jsonify({"status": "ok", "message": f"Preset '{name}' {action}", "action": action})


@app.route("/api/presets/<tab>/<name>", methods=["DELETE"])
def delete_preset(tab, name):
    """Delete a preset."""
    if tab not in ("anima", "illustrious"):
        return jsonify({"status": "error", "message": "Tab must be 'anima' or 'illustrious'"}), 400

    presets = _load_presets()
    if name not in presets.get(tab, {}):
        return jsonify({"status": "error", "message": f"Preset '{name}' not found"}), 404

    del presets[tab][name]
    _save_presets(presets)
    return jsonify({"status": "ok", "message": f"Preset '{name}' deleted"})


# ─── LoRA Compression (SVD Resize) ────────────────────────────────────────

def _resolve_lora_path(path):
    """If path is a folder, find the first .safetensors file inside it."""
    if os.path.isdir(path):
        safetensors_files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
        if not safetensors_files:
            return None, "No .safetensors files found in folder"
        if len(safetensors_files) > 1:
            names = ", ".join(safetensors_files)
            return None, f"Multiple .safetensors files found: {names}. Please specify the exact file."
        return os.path.join(path, safetensors_files[0]), None
    return path, None

@app.route("/api/compress-lora", methods=["POST"])
def compress_lora():
    """Compress a LoRA file to a smaller rank using SVD decomposition."""
    import numpy as np
    from safetensors.numpy import load_file, save_file

    data = request.json
    input_path = data.get("input_path", "").strip()
    new_rank = int(data.get("new_rank", 8))
    save_precision = data.get("precision", "bf16")

    # Auto-resolve folder to file
    input_path, err = _resolve_lora_path(input_path)
    if err:
        return jsonify({"status": "error", "message": err}), 400

    if not input_path or not os.path.isfile(input_path):
        return jsonify({"status": "error", "message": "LoRA file not found. Provide a path to a .safetensors file or a folder containing one."}), 400

    if not input_path.endswith(".safetensors"):
        return jsonify({"status": "error", "message": "Only .safetensors files are supported"}), 400

    try:
        input_size = os.path.getsize(input_path)
        state_dict = load_file(input_path)

        # Group lora_up and lora_down pairs
        pairs = {}
        other_keys = {}

        for key, tensor in state_dict.items():
            if ".lora_up." in key:
                base = key.replace(".lora_up.weight", "")
                pairs.setdefault(base, {})["up"] = tensor
                pairs[base]["up_key"] = key
            elif ".lora_down." in key:
                base = key.replace(".lora_down.weight", "")
                pairs.setdefault(base, {})["down"] = tensor
                pairs[base]["down_key"] = key
            else:
                other_keys[key] = tensor

        new_state = dict(other_keys)
        compressed_count = 0
        skipped_count = 0

        # Detect original dtype from the first LoRA weight
        orig_dtype = np.float32  # default
        for base, tensors in pairs.items():
            if "up" in tensors:
                orig_dtype = tensors["up"].dtype
                break

        # Determine output dtype: preserve original unless user explicitly chose to downcast
        if save_precision == "fp16":
            out_dtype = np.float16
        elif save_precision == "fp32":
            out_dtype = np.float32
        else:
            # "bf16" or "original" — preserve original dtype
            out_dtype = orig_dtype

        for base, tensors in pairs.items():
            if "up" not in tensors or "down" not in tensors:
                # Incomplete pair — keep as-is
                if "up" in tensors:
                    new_state[tensors["up_key"]] = tensors["up"]
                if "down" in tensors:
                    new_state[tensors["down_key"]] = tensors["down"]
                skipped_count += 1
                continue

            up_raw = tensors["up"].astype(np.float32)
            down_raw = tensors["down"].astype(np.float32)

            # ── Handle Conv2d layers (4D tensors) ──
            # SDXL/Illustrious LoRAs have conv layers with shapes like:
            #   up:   (out_ch, rank, 1, 1)     — always 1×1 kernel
            #   down: (rank, in_ch, kH, kW)    — can be 1×1 or 3×3
            # Anima LoRAs are always 2D — this code is a no-op for them.
            up_shape = up_raw.shape
            down_shape = down_raw.shape
            is_conv = up_raw.ndim == 4

            if is_conv:
                # Flatten to 2D for SVD
                up = up_raw.reshape(up_shape[0], up_shape[1])       # (out_ch, rank)
                down = down_raw.reshape(down_shape[0], -1)          # (rank, in_ch*kH*kW)
            else:
                up = up_raw    # (out_features, rank)
                down = down_raw  # (rank, in_features)

            current_rank = down.shape[0]
            target_rank = min(new_rank, current_rank)

            if target_rank >= current_rank:
                # Already at or below target rank — keep as-is
                new_state[f"{base}.lora_up.weight"] = tensors["up"]
                new_state[f"{base}.lora_down.weight"] = tensors["down"]
                skipped_count += 1
                continue

            # ── Optimized SVD via QR factorization ──
            # Instead of SVD on full (out×in) matrix, decompose via thin QR
            # and only SVD the tiny (rank×rank) core. ~1000x faster.
            Q_up, R_up = np.linalg.qr(up)        # Q_up (out, r), R_up (r, r)
            Q_dn, R_dn = np.linalg.qr(down.T)    # Q_dn (in, r),  R_dn (r, r)

            # SVD of tiny (rank × rank) core matrix — instant
            M = R_up @ R_dn.T                     # (r, r) — e.g. 32×32
            U_m, S, Vt_m = np.linalg.svd(M, full_matrices=False)

            # Truncate to target rank
            U_k = U_m[:, :target_rank]
            S_k = S[:target_rank]
            Vt_k = Vt_m[:target_rank, :]

            # Rebuild compressed LoRA factors
            sqrt_S = np.sqrt(S_k)
            new_up_mat = (Q_up @ U_k) * sqrt_S[np.newaxis, :]   # (out, new_rank)
            new_dn_mat = (sqrt_S[:, np.newaxis] * Vt_k) @ Q_dn.T  # (new_rank, in)

            if is_conv:
                # Reshape back to 4D preserving original spatial dims
                new_up_mat = new_up_mat.reshape(up_shape[0], target_rank, 1, 1)
                new_dn_mat = new_dn_mat.reshape(target_rank, down_shape[1], down_shape[2], down_shape[3])

            new_state[f"{base}.lora_up.weight"] = new_up_mat.astype(out_dtype)
            new_state[f"{base}.lora_down.weight"] = new_dn_mat.astype(out_dtype)
            compressed_count += 1

        # Scale alpha values proportionally to preserve effective weight
        for key in list(new_state.keys()):
            if ".alpha" in key:
                base = key.replace(".alpha", "")
                if base in pairs and "down" in pairs[base]:
                    old_rank = pairs[base]["down"].shape[0]
                    if old_rank > new_rank:
                        old_alpha = float(new_state[key].item())
                        new_alpha = old_alpha * new_rank / old_rank
                        new_state[key] = np.array(new_alpha, dtype=new_state[key].dtype)

        # Generate output filename
        input_name = Path(input_path).stem
        out_name = f"{input_name}_r{new_rank}.safetensors"
        output_path = str(Path(input_path).parent / out_name)

        save_file(new_state, output_path)
        output_size = os.path.getsize(output_path)

        ratio = (1 - output_size / input_size) * 100

        return jsonify({
            "status": "ok",
            "input_size_mb": round(input_size / (1024 * 1024), 1),
            "output_size_mb": round(output_size / (1024 * 1024), 1),
            "compression_ratio": round(ratio, 1),
            "layers_compressed": compressed_count,
            "layers_skipped": skipped_count,
            "output_path": output_path,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/compress-lora/info", methods=["POST"])
def compress_lora_info():
    """Get info about a LoRA file (current rank, size, layer count)."""
    import numpy as np
    from safetensors.numpy import load_file

    data = request.json
    input_path = data.get("input_path", "").strip()

    # Auto-resolve folder to file
    input_path, err = _resolve_lora_path(input_path)
    if err:
        return jsonify({"status": "error", "message": err}), 400

    if not input_path or not os.path.isfile(input_path):
        return jsonify({"status": "error", "message": "File not found. Provide a path to a .safetensors file or a folder containing one."}), 400

    try:
        file_size = os.path.getsize(input_path)
        state_dict = load_file(input_path)

        # Detect current rank from lora_down layers
        ranks = set()
        lora_layer_count = 0
        alpha_values = set()

        for key, tensor in state_dict.items():
            if ".lora_down." in key:
                ranks.add(tensor.shape[0])
                lora_layer_count += 1
            if ".alpha" in key:
                alpha_values.add(int(float(tensor.item())))

        # Detect model type from key patterns
        model_type = "Unknown"
        for key in state_dict.keys():
            # Anima uses "lora_unet_blocks_X_self_attn/cross_attn/mlp" (DiT) and "lora_te_layers_X" (Qwen)
            if "lora_unet_blocks_" in key and ("_self_attn_" in key or "_cross_attn_" in key):
                model_type = "Anima (DiT)"
                break
            if "lora_te_layers_" in key and "_mlp_" in key:
                model_type = "Anima (DiT)"
                break
            # SDXL/Illustrious uses "lora_unet_input_blocks/output_blocks/mid_block" and "lora_te1_/lora_te2_"
            if "lora_unet_input_blocks" in key or "lora_unet_output_blocks" in key or "lora_unet_mid_block" in key:
                model_type = "SDXL/Illustrious"
                break
            if "lora_te1_" in key or "lora_te2_" in key:
                model_type = "SDXL/Illustrious"
                break

        return jsonify({
            "status": "ok",
            "file_size_mb": round(file_size / (1024 * 1024), 1),
            "current_rank": sorted(list(ranks)),
            "alpha_values": sorted(list(alpha_values)),
            "lora_layers": lora_layer_count,
            "total_keys": len(state_dict),
            "model_type": model_type,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket

    PORTS_TO_TRY = [9005, 9006, 9007, 9008, 9009]

    chosen_port = None
    for port in PORTS_TO_TRY:
        try:
            # Quick check: can we actually bind this port?
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.bind(("0.0.0.0", port))
            test_sock.close()
            chosen_port = port
            break
        except OSError:
            print(f"  [!] Port {port} is blocked, trying next...")

    if chosen_port is None:
        print("  [ERROR] All ports are blocked! Try restarting your computer.")
        input("Press Enter to exit...")
        sys.exit(1)

    print("=" * 60)
    print("  Anima & Illustrious LoRA Training Web UI")
    print(f"  Open: http://localhost:{chosen_port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=chosen_port, debug=True, use_reloader=False)
