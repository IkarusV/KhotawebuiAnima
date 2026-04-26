# KhotawebuiAnima

Flask Web UI for training Anima and Illustrious LoRAs with kohya sd-scripts locally or on Modal cloud GPUs.

## What is included

- Flask web app: `app.py`
- UI templates and CSS: `templates/`, `static/`
- Modal cloud training scripts: `modal/train_anima_modal.py`, `modal/upload_models.py`, `modal/upscale_modal.py`
- Bundled kohya sd-scripts source for local training: `sd-scripts-main/`
- Windows launcher: `start.bat`

Large runtime artifacts are intentionally excluded from GitHub: virtual environments, datasets, downloaded models, generated LoRA outputs, generated Modal job files, history, and sd-scripts runtime caches.

## Requirements

- Windows
- Python 3.10 or newer
- A Modal account for cloud GPU training
- Optional local training: a kohya-ss/sd-scripts checkout with its own working environment

## Quick Start

1. Download or clone this repository.
2. Double-click `start.bat`.
3. The launcher creates `venv`, installs requirements, prepares the bundled sd-scripts environment for local training, and starts the Web UI.
4. Open the printed local URL, usually `http://localhost:9005`.

To run manually:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

## Modal Setup

Modal credentials are personal secrets. Do not commit them.

After the venv is installed, authenticate Modal with either the Web UI account panel or the command line:

```bat
venv\Scripts\python -m modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET
```

The app uses the Modal volume named `anima-training-data` for models, datasets, and outputs. Modal GPU work can cost money, so confirm your Modal account and GPU choice before launching jobs.

To upload the default Anima model files into the Modal volume:

```bat
venv\Scripts\python -m modal run modal/upload_models.py
```

## Local sd-scripts Setup

The app includes `sd-scripts-main` and defaults local training to that folder. On first run, `start.bat` creates `sd-scripts-main\venv`, installs PyTorch, and installs the sd-scripts requirements.

That first local-training setup can take a while because PyTorch is large. You can still set a different sd-scripts path from the Web UI settings panel if you prefer an external checkout.

## GitHub Safety

This repo should not contain:

- `venv/`
- `datasets/`
- `history/`
- `output/`
- `downloads/`
- `upscaled_datasets/`
- `sd-scripts-main/venv/`
- `sd-scripts-main/logs/`
- `sd-scripts-main/build/`
- `modal/_run_*.py`
- `modal/_config_*.json`
- model files such as `.safetensors`, `.ckpt`, `.pt`, `.pth`
- Modal tokens, API keys, `.env` files, or `settings.json`
