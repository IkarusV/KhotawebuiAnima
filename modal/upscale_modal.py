"""
AI Image Upscaler on Modal Cloud GPUs.
Uses Real-ESRGAN to 4x upscale images, then resize to target Anima bucket resolutions.
Aspect-ratio aware: square→1024x1024, portrait→832x1216, landscape→1216x832.
"""

import modal
import os

app = modal.App("anima-upscaler")

vol = modal.Volume.from_name("anima-training-data", create_if_missing=True)
VOL_MOUNT = "/data"

# Image with Real-ESRGAN and PIL
upscale_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "wget")
    .run_commands(
        "pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "realesrgan",
        "basicsr",
        "gfpgan",
        "opencv-python==4.10.0.84",
        "Pillow",
        "numpy",
    )
    .run_commands(
        # Download Real-ESRGAN x4plus model weights
        "mkdir -p /models && "
        "wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth "
        "-O /models/RealESRGAN_x4plus.pth",
        # Download anime-specific model
        "wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth "
        "-O /models/RealESRGAN_x4plus_anime_6B.pth",
        # Download 4x-UltraSharp from HuggingFace
        "pip install huggingface-hub && "
        "python -c \"from huggingface_hub import hf_hub_download; "
        "hf_hub_download('lokCX/4x-Ultrasharp', '4x-UltraSharp.pth', local_dir='/models')\"",
    )
)


def get_target_resolution(width, height):
    """Determine target resolution based on aspect ratio.
    Square → 1024x1024, Portrait → 832x1216, Landscape → 1216x832.
    """
    ratio = width / height
    if 0.9 <= ratio <= 1.1:
        # Square
        return 1024, 1024
    elif ratio < 0.9:
        # Portrait (taller than wide)
        return 832, 1216
    else:
        # Landscape (wider than tall)
        return 1216, 832


def resize_to_target(img, target_w, target_h):
    """Center-crop and resize an image to target dimensions."""
    from PIL import Image

    w, h = img.size
    target_ratio = target_w / target_h
    src_ratio = w / h

    if src_ratio > target_ratio:
        # Source is wider — crop sides
        crop_h = h
        crop_w = int(crop_h * target_ratio)
        crop_x = (w - crop_w) // 2
        crop_y = 0
    else:
        # Source is taller — crop top/bottom
        crop_w = w
        crop_h = int(crop_w / target_ratio)
        crop_x = 0
        crop_y = (h - crop_h) // 2

    cropped = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    resized = cropped.resize((target_w, target_h), Image.LANCZOS)
    return resized


@app.function(
    image=upscale_image,
    gpu="H100",
    volumes={VOL_MOUNT: vol},
    timeout=600,
)
def upscale_images(dataset_name: str, model_type: str = "anime"):
    """Upscale all images in upscale_input/<dataset_name>/ and save to upscale_output/<dataset_name>/."""
    import cv2
    import numpy as np
    from PIL import Image

    # Fix basicsr compatibility with torchvision >= 0.18
    # (functional_tensor was removed, its contents moved to functional)
    import sys
    import torchvision.transforms.functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    input_dir = f"{VOL_MOUNT}/upscale_input/{dataset_name}"
    output_dir = f"{VOL_MOUNT}/upscale_output/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Select model architecture
    if model_type == "anime":
        model_path = "/models/RealESRGAN_x4plus_anime_6B.pth"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    elif model_type == "ultrasharp":
        model_path = "/models/4x-UltraSharp.pth"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    else:
        model_path = "/models/RealESRGAN_x4plus.pth"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # Load weights manually (different models use different checkpoint formats)
    import torch
    import re
    loadnet = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'params_ema' in loadnet:
        state_dict = loadnet['params_ema']
    elif 'params' in loadnet:
        state_dict = loadnet['params']
    else:
        state_dict = loadnet

    # Convert old ESRGAN key format → new basicsr RRDBNet format
    # (4x-UltraSharp and similar community models use the old format)
    first_key = next(iter(state_dict))
    if first_key.startswith('model.'):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k == 'model.0.weight':
                new_state_dict['conv_first.weight'] = v
            elif k == 'model.0.bias':
                new_state_dict['conv_first.bias'] = v
            elif k.startswith('model.1.sub.'):
                # model.1.sub.{block}.RDB{n}.conv{m}.0.{w/b}
                # → body.{block}.rdb{n}.conv{m}.{w/b}
                rest = k.replace('model.1.sub.', '')
                parts = rest.split('.')
                block_idx = int(parts[0])
                if parts[1].startswith('RDB'):
                    rdb_n = parts[1][3:]  # "1", "2", "3"
                    conv_m = parts[2][4:]  # conv1 → "1"
                    wb = parts[4]  # "weight" or "bias"
                    new_key = f'body.{block_idx}.rdb{rdb_n}.conv{conv_m}.{wb}'
                    new_state_dict[new_key] = v
                else:
                    # model.1.sub.23.weight → conv_body.weight (the trunk_conv)
                    new_state_dict[f'conv_body.{parts[1]}'] = v
            elif k == 'model.3.weight':
                new_state_dict['conv_up1.weight'] = v
            elif k == 'model.3.bias':
                new_state_dict['conv_up1.bias'] = v
            elif k == 'model.6.weight':
                new_state_dict['conv_up2.weight'] = v
            elif k == 'model.6.bias':
                new_state_dict['conv_up2.bias'] = v
            elif k == 'model.8.weight':
                new_state_dict['conv_hr.weight'] = v
            elif k == 'model.8.bias':
                new_state_dict['conv_hr.bias'] = v
            elif k == 'model.10.weight':
                new_state_dict['conv_last.weight'] = v
            elif k == 'model.10.bias':
                new_state_dict['conv_last.bias'] = v
        state_dict = new_state_dict
        print(f"Converted old ESRGAN format → new RRDBNet format ({len(state_dict)} keys)")

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Monkey-patch torch.load so RealESRGANer.__init__ gets our converted weights
    # (RealESRGANer doesn't support model_path=None and always reloads from disk)
    _original_torch_load = torch.load
    torch.load = lambda *a, **kw: {'params': state_dict}

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,  # Use fp16 for speed
    )

    torch.load = _original_torch_load  # Restore

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted([f for f in os.listdir(input_dir)
                    if os.path.splitext(f)[1].lower() in image_exts])

    results = []
    for i, fname in enumerate(files):
        input_path = os.path.join(input_dir, fname)
        print(f"[{i+1}/{len(files)}] Upscaling {fname}...")

        # Read with OpenCV
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  Skipping {fname} — could not read")
            continue

        orig_h, orig_w = img.shape[:2]

        # 4x upscale with Real-ESRGAN
        output, _ = upsampler.enhance(img, outscale=4)
        upscaled_h, upscaled_w = output.shape[:2]

        # Convert to PIL for resize
        if len(output.shape) == 2:
            pil_img = Image.fromarray(output)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

        # Determine target resolution based on ORIGINAL aspect ratio
        target_w, target_h = get_target_resolution(orig_w, orig_h)

        # Resize to target
        final = resize_to_target(pil_img, target_w, target_h)

        # Save as PNG
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(output_dir, out_name)
        final.save(out_path, "PNG")

        results.append({
            "filename": fname,
            "original": f"{orig_w}x{orig_h}",
            "upscaled_4x": f"{upscaled_w}x{upscaled_h}",
            "final": f"{target_w}x{target_h}",
        })
        print(f"  {orig_w}x{orig_h} → 4x → {upscaled_w}x{upscaled_h} → {target_w}x{target_h}")

    # Also copy over any .txt files (captions)
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    for txt in txt_files:
        import shutil
        shutil.copy2(os.path.join(input_dir, txt), os.path.join(output_dir, txt))

    vol.commit()

    return {
        "status": "ok",
        "dataset_name": dataset_name,
        "total": len(results),
        "captions_copied": len(txt_files),
        "details": results,
    }
