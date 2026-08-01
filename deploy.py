"""Modal deploy entry for sensenova-u1.

Deploy:
  modal deploy deploy.py

One unified SenseNova-U1 checkpoint (NEO-unify architecture, no VE/VAE) serves
three node slots:
  - image-gen      → infographic / text-to-image
  - image-edit     → instruction-guided image editing
  - image-gen-text → visual understanding (VQA / captioning)

Design constraints:
  - Keep this file mostly self-contained because Modal remote imports may mount
    only the entry file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal
from tongflow import deploy
from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.models.image_gen_text import ImageGenTextInput, ImageGenTextOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import node_slot


_cfg: dict[str, Any] = {}
_hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
REPO_ID = str(_hf.get("repoId") or "sensenova/SenseNova-U1-8B-MoT-Infographic")
MODEL_DIR = f"/models/{REPO_ID}"

# 8-step distilled acceleration LoRA. Merged into the base weights once at load
# time (in-place, global — shared by every slot). Set LORA_PATH = None to fall
# back to the 50-step base path; then also restore DEFAULT_NUM_STEPS = 50 and
# DEFAULT_CFG_SCALE = 4.0 below so the sampling params match the active weights.
LORA_FILENAME = "SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0.safetensors"
LORA_PATH: str | None = f"/models/loras/{LORA_FILENAME}"

# Diffusion / sampling defaults — plugin-internal, not part of the ABI contract.
# Values track the active weights: 8-step distilled LoRA (num_steps=8, cfg=1.0).
GRID_FACTOR = 32  # output H/W must be a multiple of this (= patch_size / downsample).
DEFAULT_WIDTH = 2048  # SUPPORTED_RESOLUTIONS["1:1"]
DEFAULT_HEIGHT = 2048
DEFAULT_SEED = 42
DEFAULT_NUM_STEPS = 8
DEFAULT_CFG_SCALE = 1.0
DEFAULT_IMG_CFG_SCALE = 1.0  # 1.0 = image CFG disabled (editing only).
DEFAULT_CFG_NORM = "none"
DEFAULT_TIMESTEP_SHIFT = 3.0
DEFAULT_TARGET_PIXELS = 2048 * 2048  # editing output pixel budget when size is implicit.
DEFAULT_INPUT_MAX_PIXELS = 2048 * 2048  # editing input pixel budget (aspect preserved).
# "sdpa" avoids a flash-attn build at image-bake time; "full" keeps weights on GPU.
ATTN_BACKEND = "sdpa"
VRAM_MODE = "full"

# Prompt auto-enhancement (image-gen only). The model renders text correctly only
# when every string is spelled out in the prompt; a short user idea otherwise gets
# gibberish labels. So U1 first rewrites a short prompt into a structured prompt
# (using its own LLM — no external API), then generates. A long prompt (the user
# already wrote detail) or one that already quotes text is passed through untouched.
# Set ENABLE_PROMPT_ENHANCE = False to disable entirely.
#
# The system prompt is visual-first on purpose: the packaged "infographic" template
# maximizes information density (long body paragraphs), which both crowds out
# imagery and overwhelms the 8-step LoRA's text rendering. This rewrite keeps the
# critical "quote every rendered string" rule but pushes for many illustrations /
# icons / charts and only sparse text — which also renders far cleaner at 8 steps.
ENABLE_PROMPT_ENHANCE = True
ENHANCE_MAX_INPUT_CHARS = 300
ENHANCE_MAX_NEW_TOKENS = 1024
ENHANCE_SYSTEM_PROMPT = """\
# Role
You are a senior visual designer who turns a user's [Raw Idea] into a VISUAL-FIRST \
infographic image-generation prompt. Your infographics communicate through imagery \
— illustrations, icons, charts, and diagrams — with only minimal supporting text.

# Task
Rewrite the user's [Raw Idea] into a single image-generation prompt (about 200-320 \
words) describing an infographic that is IMAGE-HEAVY and TEXT-LIGHT.

# Core Principles
1. Imagery over text. Spend most of the description on concrete visual elements: \
detailed illustrations, pictograms/icons, charts (bar, pie, flow), diagrams, and \
visual metaphors. Each key point is carried by a picture, not a paragraph.
2. Minimal text, rendered LARGE. Include ONLY a short title plus at most about six \
short labels or numbers across the whole image. Do NOT write body paragraphs, long \
sentences, fine print, or small captions. Every text element must be specified as \
large and bold (e.g. "the large bold title", "a big bold label"); the model renders \
big text cleanly but garbles small text, so there must be no small text anywhere. \
Prefer single words or two-word phrases.
3. Concrete icons (CRITICAL). Describe the exact visual content of every icon or \
illustration so it matches its label, e.g. "a detailed illustration of a yellow sun \
with wavy heat lines over blue ocean water". Never write "an icon" or "a graphic" \
generically.
4. Layout & style. Specify the overall layout (e.g. central diagram with \
surrounding callouts, horizontal flow, grid), a clear visual style, a background \
texture, and a harmonious named color palette. Use generous whitespace.

# Text Rendering Protocol
- Every piece of text meant to appear in the image MUST be wrapped in quotes.
- Keep quoted text short — titles, headers, single words, short phrases, numbers. \
Avoid full sentences.
- Preserve every proper noun, number, and date from the [Raw Idea] exactly.
- NEVER put quotes around style, layout, color, or icon descriptions.

# Constraints
- Match the language of the [Raw Idea] (Chinese in -> Chinese out; English in -> \
English out).
- Use descriptive color names, never hex codes.
- Start immediately with the visual description. No preamble, no meta-commentary.

Now rewrite the user's [Raw Idea] into the visual-first infographic prompt:"""

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)


# ── app ──────────────────────────────────────────────────────────────────────

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")
    # CUDA 12.8 torch wheels first so the sensenova-u1 install below reuses them
    # instead of resolving CPU torch from PyPI.
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("tongflow==0.2.20")
    # sensenova-u1 is not on PyPI; install from source. A `pip install git+...`
    # forces `git submodule update --recursive`, which fails on the repo's broken
    # `evaluation/` submodule refs — so shallow-clone WITHOUT submodules and install
    # the local path. The package code lives entirely under src/sensenova_u1 and
    # needs none of those submodules. Its pyproject pins transformers / accelerate /
    # huggingface-hub etc., which install here on top of the cu128 torch above.
    .run_commands(
        "git clone --depth 1 https://github.com/OpenSenseNova/SenseNova-U1.git /opt/sensenova-u1",
        "pip install /opt/sensenova-u1",
    )
)

with image.imports():
    import io

    import numpy as np
    import torch
    from PIL import Image

    import sensenova_u1
    from sensenova_u1.models.neo_unify.conversation import get_conv_template
    from sensenova_u1.models.neo_unify.utils import load_image_native, smart_resize
    from sensenova_u1.utils import (
        load_and_merge_lora_weight_from_safetensors,
        load_model_and_tokenizer,
        make_offload_ctx,
        vram_mode_to_prefetch_count,
    )


@deploy
@app.cls(
    scaledown_window=5,
    image=image,
    gpu="A100-80GB",
    volumes={"/models": volume},
    timeout=1200,
)
class Inference:
    @modal.enter()
    def load(self):
        sensenova_u1.set_attn_backend(ATTN_BACKEND)
        self.device = "cuda"
        self.prefetch_count = vram_mode_to_prefetch_count(VRAM_MODE)
        self.model, self.tokenizer = load_model_and_tokenizer(
            MODEL_DIR,
            dtype=torch.bfloat16,
            device=self.device,
            for_offload=self.prefetch_count > 0,
        )
        if LORA_PATH:
            # In-place merge of the 8-step distilled LoRA into the base weights.
            self.model = load_and_merge_lora_weight_from_safetensors(
                self.model, LORA_PATH
            )

        # Visual-first prompt-enhancement system prompt (see ENHANCE_SYSTEM_PROMPT).
        self._enhance_system: str | None = (
            ENHANCE_SYSTEM_PROMPT if ENABLE_PROMPT_ENHANCE else None
        )
        # Token ids required by the text-only generate() path (chat() sets these
        # too; the model asserts img_context_token_id is not None in generate()).
        self._img_context_token_id = self.tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>"
        )
        self._img_start_token_id = self.tokenizer.convert_tokens_to_ids("<img>")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _snap(self, value: int) -> int:
        """Round a side length down to a valid image-token grid multiple."""
        return max(GRID_FACTOR, (int(value) // GRID_FACTOR) * GRID_FACTOR)

    @staticmethod
    def _strip_think(text: str) -> str:
        """Drop the model's <think>...</think> reasoning, keep the final answer."""
        marker = "</think>"
        idx = text.rfind(marker)
        if idx != -1:
            text = text[idx + len(marker) :]
        return text.strip()

    def _should_enhance(self, text: str) -> bool:
        """Enhance only a short, un-quoted idea; respect detailed prompts as-is."""
        if not self._enhance_system:
            return False
        if len(text) > ENHANCE_MAX_INPUT_CHARS:
            return False
        # The user already enumerated text to render → trust their wording.
        if text.count('"') >= 2 or "“" in text:
            return False
        return True

    def _enhance_prompt(self, raw: str) -> str:
        """Rewrite a short idea into a dense, fully-quoted infographic prompt
        using U1's own text generation (no external LLM). Text-only path mirrors
        chat() minus the image: generate() handles pixel_values/grid_hw = None."""
        self.model.img_context_token_id = self._img_context_token_id
        self.model.img_start_token_id = self._img_start_token_id

        template = get_conv_template(self.model.template)
        template.system_message = self._enhance_system
        template.append_message(template.roles[0], raw)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        eos_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())

        with make_offload_ctx(self.model, self.prefetch_count, self.device) as off:
            out = off.generate(
                pixel_values=None,
                input_ids=input_ids,
                grid_hw=None,
                attention_mask=attention_mask,
                max_new_tokens=ENHANCE_MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=eos_id,
            )
        text = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        text = text.split(template.sep.strip())[0].strip()
        return text or raw

    def _to_png(self, batch) -> bytes:
        """[B, 3, H, W] normalized float tensor → PNG bytes of the first image."""
        x = batch.float()
        mean = torch.tensor((0.5, 0.5, 0.5), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor((0.5, 0.5, 0.5), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        arr = (x * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        arr = (arr * 255.0).round().astype(np.uint8)
        img = Image.fromarray(arr[0])
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _load_input_rgb(self, img_bytes: bytes, input_max_pixels: int):
        """Decode to RGB (flatten RGBA on white) and resize to a pixel budget."""
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        img = img.convert("RGB")
        resized_h, resized_w = smart_resize(
            height=img.height,
            width=img.width,
            factor=GRID_FACTOR,
            min_pixels=input_max_pixels,
            max_pixels=input_max_pixels,
        )
        if (resized_w, resized_h) != img.size:
            img = img.resize((resized_w, resized_h), Image.LANCZOS)
        return img

    def _t2i_png(self, prompt: str, width: int, height: int, seed: int) -> bytes:
        w, h = self._snap(width), self._snap(height)
        with make_offload_ctx(self.model, self.prefetch_count, self.device) as off:
            out = off.t2i_generate(
                self.tokenizer,
                prompt,
                image_size=(w, h),
                cfg_scale=DEFAULT_CFG_SCALE,
                cfg_norm=DEFAULT_CFG_NORM,
                timestep_shift=DEFAULT_TIMESTEP_SHIFT,
                cfg_interval=(0.0, 1.0),
                num_steps=DEFAULT_NUM_STEPS,
                batch_size=1,
                seed=seed,
                think_mode=False,
            )
        return self._to_png(out)

    def _edit_png(
        self,
        prompt: str,
        img_bytes: bytes,
        explicit_size: tuple[int, int] | None,
        match_input: bool,
        seed: int,
    ) -> bytes:
        pil = self._load_input_rgb(img_bytes, DEFAULT_INPUT_MAX_PIXELS)
        if explicit_size is not None:
            out_w, out_h = self._snap(explicit_size[0]), self._snap(explicit_size[1])
        elif match_input:
            out_w, out_h = self._snap(pil.width), self._snap(pil.height)
        else:
            # Match the input aspect ratio, normalize total pixels to the budget.
            resized_h, resized_w = smart_resize(
                height=pil.height,
                width=pil.width,
                factor=GRID_FACTOR,
                min_pixels=DEFAULT_TARGET_PIXELS,
                max_pixels=DEFAULT_TARGET_PIXELS,
            )
            out_w, out_h = resized_w, resized_h
        with make_offload_ctx(self.model, self.prefetch_count, self.device) as off:
            out = off.it2i_generate(
                self.tokenizer,
                prompt,
                [pil],
                image_size=(out_w, out_h),
                cfg_scale=DEFAULT_CFG_SCALE,
                img_cfg_scale=DEFAULT_IMG_CFG_SCALE,
                cfg_norm=DEFAULT_CFG_NORM,
                timestep_shift=DEFAULT_TIMESTEP_SHIFT,
                cfg_interval=(0.0, 1.0),
                num_steps=DEFAULT_NUM_STEPS,
                batch_size=1,
                think_mode=False,
                seed=seed,
            )
        return self._to_png(out)

    def _vqa(
        self,
        img_bytes: bytes,
        question: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int | None,
    ) -> str:
        # load_image_native handles RGBA flattening and its own smart-resize.
        pixel_values, grid_hw = load_image_native(Image.open(io.BytesIO(img_bytes)))
        pixel_values = pixel_values.to(self.device, dtype=self.model.dtype)
        grid_hw = grid_hw.to(self.device)

        generation_config: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p
            if top_k is not None:
                generation_config["top_k"] = top_k

        with make_offload_ctx(self.model, self.prefetch_count, self.device) as off:
            response, _ = off.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=True,
                grid_hw=grid_hw,
            )
        return self._strip_think(str(response))

    # ── slots ────────────────────────────────────────────────────────────────

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN)
    def image_gen(self, input: ImageGenInput) -> ImageGenOutput:
        text = (input.text or "").strip()
        if not text:
            return ImageGenOutput(success=False, error="Missing text prompt")
        if self._should_enhance(text):
            text = self._enhance_prompt(text)
        raw = self._t2i_png(
            text,
            width=int(input.width) if input.width is not None else DEFAULT_WIDTH,
            height=int(input.height) if input.height is not None else DEFAULT_HEIGHT,
            seed=int(input.seed) if input.seed is not None else DEFAULT_SEED,
        )
        return ImageGenOutput(success=True, image=asset(raw, mime="image/png"))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_EDIT)
    def image_edit(self, input: ImageEditInput) -> ImageEditOutput:
        text = (input.text or "").strip()
        if not text:
            return ImageEditOutput(success=False, error="Missing edit instruction")
        explicit: tuple[int, int] | None = None
        if input.width is not None and input.height is not None:
            explicit = (int(input.width), int(input.height))
        raw = self._edit_png(
            text,
            prompt_media_to_bytes(input.image),
            explicit_size=explicit,
            match_input=bool(input.match_input_size),
            seed=int(input.seed) if input.seed is not None else DEFAULT_SEED,
        )
        return ImageEditOutput(success=True, image=asset(raw, mime="image/png"))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN_TEXT)
    def image_gen_text(self, input: ImageGenTextInput) -> ImageGenTextOutput:
        if input.image is None:
            return ImageGenTextOutput(success=False, error="Missing input image")
        question = (input.text or "").strip() or "Describe this image in detail."
        if input.system:
            question = f"{input.system.strip()}\n\n{question}"
        # Sampling is engaged only when the caller supplies a sampling knob.
        do_sample = (
            input.temperature is not None
            or input.top_p is not None
            or input.top_k is not None
        )
        text = self._vqa(
            prompt_media_to_bytes(input.image),
            question,
            max_new_tokens=int(input.max_new_tokens)
            if input.max_new_tokens is not None
            else 1024,
            do_sample=do_sample,
            temperature=float(input.temperature) if input.temperature is not None else 0.7,
            top_p=float(input.top_p) if input.top_p is not None else 0.9,
            top_k=int(input.top_k) if input.top_k is not None else None,
        )
        return ImageGenTextOutput(success=True, text=text)
