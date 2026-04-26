# KhotawebuiAnima

Flask Web UI for training Anima and Illustrious LoRAs with kohya sd-scripts locally or on Modal cloud GPUs.

## What is included

- Flask web app: `app.py`
- UI templates and CSS: `templates/`, `static/`
- Modal cloud training scripts: `modal/train_anima_modal.py`, `modal/upload_models.py`, `modal/upscale_modal.py`
- Windows launcher: `start.bat`

Large runtime artifacts are intentionally excluded from GitHub: virtual environments, datasets, downloaded models, generated LoRA outputs, generated Modal job files, history, and local sd-scripts checkouts.

## Requirements

- Windows
- Python 3.10 or newer
- A Modal account for cloud GPU training
- Optional local training: a kohya-ss/sd-scripts checkout with its own working environment

## Quick Start

1. Download or clone this repository.
2. Double-click `start.bat`.
3. The launcher creates `venv`, installs requirements, and starts the Web UI.
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

The app default points local training at `sd-scripts-main` inside the repository folder. Keep that folder out of git.

Clone kohya sd-scripts beside the app files:

```bat
git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts-main
```

Then install and configure sd-scripts according to the kohya-ss project instructions. You can also set a different path from the Web UI settings panel.

## GitHub Safety

This repo should not contain:

- `venv/`
- `datasets/`
- `history/`
- `output/`
- `downloads/`
- `upscaled_datasets/`
- `sd-scripts-main/`
- `modal/_run_*.py`
- `modal/_config_*.json`
- model files such as `.safetensors`, `.ckpt`, `.pt`, `.pth`
- Modal tokens, API keys, `.env` files, or `settings.json`
