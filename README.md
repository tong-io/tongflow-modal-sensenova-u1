# tongflow-modal-sensenova-u1

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Unified multimodal generation and understanding with **SenseNova-U1** (`sensenova/SenseNova-U1-8B-MoT-Infographic`), running on a GPU via [Modal](https://modal.com). One NEO-unify checkpoint (no visual encoder, no VAE) serves all three slots.

## Capabilities

- **Image generation** (`image-gen`) — infographic / text-to-image from a prompt. Tuned for dense, text-rich layouts (posters, charts, knowledge illustrations).
- **Image editing** (`image-edit`) — instruction-guided editing of an input image.
- **Image understanding** (`image-gen-text`) — captions, Q&A, or descriptions from an image.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin downloads the weights to a Modal volume and deploys to your Modal account automatically, caching the build. The `sensenova/SenseNova-U1-8B-MoT-Infographic` weights are public — no Hugging Face token required.

## Notes

- **8-step acceleration LoRA.** The `SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0` distilled LoRA is merged into the base weights at load time, so `image-gen` and `image-edit` run at **8 steps / cfg 1.0** (~10× faster than the 50-step base). `image-gen-text` (understanding) is unaffected. Set `LORA_PATH = None` in `deploy.py` to fall back to the 50-step base path.
- The model runs on an **A100-80GB**. Default generation resolution is 2048×2048; 80 GB has ample headroom for 2K and 4K outputs. Drop to a smaller `gpu=` tier in `deploy.py` to cut cost if you only generate ≤2K.
- `sensenova-u1` is installed from source (not on PyPI) on top of CUDA 12.8 torch wheels, so the first build is slower than usual.
