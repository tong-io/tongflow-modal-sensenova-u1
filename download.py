"""Modal download entry for sensenova-u1.

Run:
  modal run download.py::download

Self-contained: Modal remote execution may mount only this file, so do not
import other local modules (e.g. `impl.py`, `config.py`).
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import modal


_cfg: dict[str, Any] = {}
_hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
REPO_ID = str(_hf.get("repoId") or "sensenova/SenseNova-U1-8B-MoT-Infographic")
REVISION = str(_hf.get("revision") or "")
MODEL_DIR = f"/models/{REPO_ID}"

# 8-step distilled acceleration LoRA (merged into the base weights at load time).
LORA_REPO_ID = "sensenova/SenseNova-U1-8B-MoT-LoRAs"
LORA_FILENAME = "SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0.safetensors"
LORA_DIR = "/models/loras"

# Must exist at module import time for Modal.
volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)

model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "huggingface_hub==1.6.0",
    ),
    volumes={"/models": volume},
    timeout=3600,
)
def _download() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    changed = False

    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"Model already exists at {MODEL_DIR}, skipping")
    else:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            revision=REVISION or None,
        )
        changed = True
        print(f"Model downloaded to {MODEL_DIR}")

    lora_target = os.path.join(LORA_DIR, LORA_FILENAME)
    if os.path.exists(lora_target):
        print(f"LoRA already exists at {lora_target}, skipping")
    else:
        os.makedirs(LORA_DIR, exist_ok=True)
        hf_hub_download(
            repo_id=LORA_REPO_ID,
            filename=LORA_FILENAME,
            local_dir=LORA_DIR,
        )
        changed = True
        print(f"LoRA downloaded to {lora_target}")

    if changed:
        volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
