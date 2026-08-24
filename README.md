<p align="center">
  <a href="https://github.com/MiniMax-AI/MiniMax-H3"><img src="https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/assets/minimax-h3-header.gif" alt="MiniMax H3"></a>
</p>

# MiniMax H3 Integrations

A community-maintained index of checkpoints, tools, and workflows for MiniMax H3, ordered by developer interest.

<a id="guides"></a>

## Official resources

* [MiniMax-H3 official model card](https://huggingface.co/MiniMaxAI/MiniMax-H3) · [official repository](https://github.com/MiniMax-AI/MiniMax-H3)
* [Video Prompt Writing Guide — Base (FL2VA)](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
* [Video Prompt Writing Guide — Reference (Ref2VA)](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
* ComfyUI [day-0 blog post](https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui) · [tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)

## Start here

This navigation guide is not a complete compatibility list.

| Goal | Start with |
| :--- | :--- |
| Run on your GPU | [Run locally](#models) — pick a stack from the [VRAM table](#recipes-vram) |
| Work with audio | [`comfyui-minimax-h3-audio-T8`](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) and [audio VAE](#components-vae) |
| Build in ComfyUI | [Official tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) · [Workflows & nodes](#nodes) |
| Write better prompts | [Prompting](#recipes-prompt) |
| Make it faster | [Speed](#speed) |
| Fine-tune it | [Training & LoRAs](#training) |
| Serve it as an API | [Serving H3](#partners) |
| Run on Apple Silicon | [`antirez/h3.c`](https://github.com/antirez/h3.c) |

<a id="models"></a>

## Run it locally

MiniMax-H3 generates video with **native stereo audio** from text, images, video, and audio inputs. It supports clips up to **2K** and **15 seconds**. Two base variants exist:

* **H3-Base-FL2VA** (first-and-last-frame mode) — accepts zero, one, or two input images. Zero images for text-to-video; one image for first- *or* last-frame-to-video; two images for first-and-last-frame-to-video.
* **H3-Base-Ref2VA** (omni-reference mode) — accepts up to **9 images, 3 video clips (2–15 s each), and 3 audio clips**, for a maximum of **12 files** total.

Checkpoints are the same size. FL2VA, trained only with keyframes, typically yields better raw output. Ref2VA accepts more reference material but has lower base quality; the [Ref Patch](#refpatch) can partially bridge this gap.

<a id="recipes-vram"></a>

### By VRAM and hardware

Find your GPU in the table, then use the notes to inform your configuration.

| Situation | Stack | Why this combination |
| :--- | :--- | :--- |
| **24 GB, first run** | `pruned_int8_convrot` DiT (19.53 GiB) + TE `nvfp4_awq` (14.61 GiB) + [`ComfyUI-MiniMaxH3-Easy`](https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy) | Easy routes T2V, I2V, first/last-frame, and reference input through a single `Media` port. Sampling, LoRAs, and decoding remain outside the node for later modification. |
| **24 GB, want speed** | The above + [TE-Speed-MiniMaxH3](https://github.com/tl2012tl/TE-Speed-MiniMaxH3) + Turbo `v4_step600_ema` at **6–8 steps** | Block-cache acceleration; v3.2 adapts to current ComfyUI's block prefetch and no longer patches core files. 6–8 steps reduce Turbo motion smear. |
| **12–16 GB** | Pruned `Q4_K_M` GGUF (10.64 GiB) or pruned `nvfp4` (11.67 GiB) + TE `Q2_K` (7.91 GiB) + fp8mix VAE pair | GGUF offers the most size options, beneficial for tight memory. `IQ1_S` is smaller at 3.78 GiB, but quality noticeably drops. |
| **8 GB** | [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) NF4 path | The project states 8 GB as its minimum for this path. Offloading performs most work here; expect slow performance, not just small memory footprint. |
| **RTX 50-series / Blackwell** | [NVIDIA Sol-Attn](https://github.com/kijai/ComfyUI-SolAttn_triton) | **1.14–1.44×** faster than SageAttention with **−37 %** MLP peak VRAM, measured on a 5090. SM89–SM121, Triton 3.6.0. Also unlocks Blackwell-only hybrid-NVFP4 checkpoints. |
| **Multi-shot / long video** | [`ComfyUI-H3-Motion-Context`](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) | H3 generates in blocks up to 15 s. Motion-Context feeds the previous block's final frame **and** audio forward, preserving motion direction and speed. |
| **Storyboard / timeline** | [`ComfyUI_MiniMaxH3_Director`](https://github.com/huangserva/ComfyUI_MiniMaxH3_Director) | Five importable templates: t2v, fl2v, r2v, v2v, and rv2v. |
| **Inpaint / local edit** | [`scraed/LanPaint`](https://github.com/scraed/LanPaint) | v2.1.0 fixed H3 support. Training-free video **and** audio inpainting. |
| **Apple Silicon** | [`antirez/h3.c`](https://github.com/antirez/h3.c) (MIT, Metal-native) | h3.c supports T2V/A, first-last-frame, and ordered Ref2VA references end-to-end, with M3 Max / M5 Max performance optimization ongoing. |
| **One-command local** | [`open-video-ai/open-video`](https://github.com/open-video-ai/open-video) | "Ollama for video models" — `install` · `pull` · `run`. |

<a id="checkpoints"></a>

### Checkpoints

| Source | What it is | Files | Total |
| :--- | :--- | :---: | ---: |
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Original diffusers weights — `transformer/` (FL2VA) and `transformer_ref/` (Ref2VA) at 14 shards / 61.73 GiB each, plus text encoder, video VAE, audio VAE, and self-contained `FL2VA/` and `Ref2VA/` pipeline folders | 280 | 464.2 GiB |
| [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | ComfyUI-repackaged single-file weights — 10 diffusion models, 3 text encoders, video + audio VAE | 17 | 433.2 GiB |

| Variant | Name | Precision | Size | Download |
| :--- | :--- | :---: | :---: | :---: |
| FL2VA | `minimax_h3_fl2va` | ![bf16][badge-bf16] | 61.73 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors) |
| FL2VA | `minimax_h3_fl2va` | ![int8][badge-int8] | 31.70 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors) |
| FL2VA | `minimax_h3_fl2va_pruned` | ![bf16][badge-bf16] | 37.46 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors) |
| FL2VA | `minimax_h3_fl2va_pruned` | ![fp8][badge-fp8] | 19.52 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors) |
| FL2VA | `minimax_h3_fl2va_pruned` | ![int8][badge-int8] | 19.53 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) |
| Ref2VA | `minimax_h3_ref2va` | ![bf16][badge-bf16] | 61.73 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_bf16.safetensors) |
| Ref2VA | `minimax_h3_ref2va` | ![int8][badge-int8] | 31.70 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors) |
| Ref2VA | `minimax_h3_ref2va_pruned` | ![bf16][badge-bf16] | 37.46 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors) |
| Ref2VA | `minimax_h3_ref2va_pruned` | ![fp8][badge-fp8] | 19.52 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors) |
| Ref2VA | `minimax_h3_ref2va_pruned` | ![int8][badge-int8] | 19.53 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) |

**"Pruned"** means AdaLN-pruned. These models are roughly 40% smaller, work only in ComfyUI, and underpin most of the consumer-GPU quants below. For a 24 GB card, `pruned_int8_convrot` (**19.53 GiB**) is the usual starting point.

<a id="quants"></a>

### Quantized Models

MiniMax provides original BF16 checkpoints. The files below are community conversions and repackaged variants, not official MiniMax releases. Confirm compatibility with your runtime's documentation before downloading.

<details>
<summary><b>Community FL2VA conversions</b></summary>

| Pruned | Precision | Method | Size | Download |
| :---: | :---: | :--- | :---: | :--- |
| | ![bf16][badge-bf16] | BF16 | 61.73 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors) |
| | ![bf16][badge-bf16] | Hybrid (fl2va base + ref2va adaln b15-49) | 20.97 GiB | [![][gh-smhfacct]](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models/resolve/main/minimax_h3_hybrid_fl2va_ref2va_b15-49.safetensors) |
| | ![bf16][badge-bf16] | Hybrid (fl2va base + ref2va adaln b20-49) | 20.97 GiB | [![][gh-smhfacct]](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models/resolve/main/minimax_h3_hybrid_fl2va_ref2va_b20-49.safetensors) |
| | ![bf16][badge-bf16] | Hybrid (fl2va base + ref2va adaln b25-49) | 20.97 GiB | [![][gh-smhfacct]](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models/resolve/main/minimax_h3_hybrid_fl2va_ref2va_b25-49.safetensors) |
| | ![bf16][badge-bf16] | Hybrid (fl2va base + ref2va adaln b30-49) | 20.97 GiB | [![][gh-smhfacct]](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models/resolve/main/minimax_h3_hybrid_fl2va_ref2va_b30-49.safetensors) |
| | ![int8][badge-int8] | ConvRot | 31.70 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors) |
| | ![int8][badge-int8] | ConvRot Lean (HQ) | 21.91 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/FL2VA/MiniMax-H3_FL2VA-INT8-ConvRot-HQ.safetensors) |
| | ![int8][badge-int8] | ConvRot | 20.94 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/FL2VA/MiniMax-H3_FL2VA-INT8-ConvRot.safetensors) |
| | ![int8][badge-int8] | ConvRot Lite | 20.33 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/FL2VA/MiniMax-H3_FL2VA-INT8-ConvRot-Lite.safetensors) |
| | ![fp8][badge-fp8] | FP8 E4M3FN | 43.78 GiB | [![][gh-rzgar]](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/minimax_h3_fl2va_fp8_e4m3fn.safetensors) |
| | ![mxfp8][badge-mxfp8] | MXFP8 | 44.34 GiB | [![][gh-rzgar]](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/minimax_h3_fl2va_mxfp8.safetensors) |
| | ![fp8][badge-fp8] | FP8 + FP16 attention | 26.70 GiB | [![][gh-rzgar]](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/minimax_h3_fl2va_fp16attn_fp8.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 (HQ) | 13.60 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/FL2VA/MiniMax-H3_FL2VA-NVFP4-HQ.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 | 10.86 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/FL2VA/MiniMax-H3_FL2VA-NVFP4.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 | 32.05 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4/resolve/main/minimax_h3_fl2va_nvfp4.safetensors) |
| | ![int4][badge-int4] | NF4 (DiffSynth) | 15.98 GiB | [![][gh-DiffSynth-Studio]](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4/resolve/main/minimax-h3-fl2va-nf4.safetensors) |
| | | OrbitQuant W4A4 | 17.03 GiB | [![][gh-WaveCut]](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/transformer/diffusion_pytorch_model-00001-of-00005.safetensors) |
| | ![int8][badge-int8] | ⚠️ DT-sQKV ConvRot | 21.00 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV/resolve/main/FL2VA/MiniMax-H3_FL2VA-DT-sQKV-INT8-ConvRot.safetensors) |
| | ![int8][badge-int8] | ⚠️ DT-sQKV ConvRot Lean | 27.99 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV/resolve/main/FL2VA/MiniMax-H3_FL2VA-DT-sQKV-INT8-ConvRot-HQ.safetensors) |
| ✓ | ![bf16][badge-bf16] | BF16 | 37.46 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors) |
| ✓ | ![fp8][badge-fp8] | FP8 scaled | 19.52 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors) |
| ✓ | ![int8][badge-int8] | ConvRot | 19.53 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) ┊ [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors) |
| ✓ | ![nvfp4][badge-nvfp4] | NVFP4 | 18.69 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4/resolve/main/minimax_h3_fl2va_pruned_nvfp4.safetensors) |
| ✓ | ![nvfp4][badge-nvfp4] | NVFP4 + ConvRot INT8 | 18.69 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4/resolve/main/minimax_h3_fl2va_pruned_nvfp4_convrot_int8.safetensors) |
| ✓ | ![nvfp4][badge-nvfp4] | NVFP4 | 11.67 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors) |
| ✓ | ![int4][badge-int4] | Mixed INT4/INT8 ConvRot | 14.81 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors) ┊ [![][gh-tsolful]](https://huggingface.co/tsolful/Minimax_H3_INT4MixedConvRot/resolve/main/minimax_h3_fl2va_pruned_INT4BQ.safetensors) |
| ✓ | ![int4][badge-int4] | Mixed INT4/INT8 ConvRot Lean | 17.27 GiB | [![][gh-tsolful]](https://huggingface.co/tsolful/Minimax_H3_INT4MixedConvRot/resolve/main/minimax_h3_fl2va_pruned_INT4Q.safetensors) |
| ✓ | ![int4][badge-int4] | INT4 ConvRot | 15.67 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4/resolve/main/minimax_h3_fl2va_pruned_int4_convrot_simple.safetensors) |
| ✓ | ![int4][badge-int4] | Mixed INT4/INT8 ConvRot | 18.92 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4/resolve/main/minimax_h3_fl2va_pruned_mixed_int4_int8_convrot_simple.safetensors) |
| ✓ | ![int4][badge-int4] | INT4 ConvRot | 16.67 GiB | [![][gh-Merserk]](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot) |
| ✓ | ![int4][badge-int4] | INT4 ConvRot (pruned) | 10.56 GiB | [![][gh-Merserk]](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot) |
| ✓ | ![int4][badge-int4] | W4A8 ConvRot | 11.68 GiB | [![][gh-AX1Y2JP]](https://huggingface.co/AX1Y2JP/MiniMax-H3-W4A8-ConvRot/resolve/main/minimax_h3_fl2va_pruned_symw4a8convrot.safetensors) ┊ [![][gh-Kijai]](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors) ┊ [![][gh-Winnougan]](https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI/resolve/main/minimax_h3_fl2va_pruned-w4a8_convrot_pruned.safetensors) |

*GGUF quants — see the [GGUF section](#gguf).*

</details>

<details>
<summary><b>Community Ref2VA conversions</b></summary>

| Pruned | Precision | Method | Size | Download |
| :---: | :---: | :--- | :---: | :--- |
| | ![bf16][badge-bf16] | BF16 | 61.73 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_bf16.safetensors) |
| | ![int8][badge-int8] | ConvRot ┊ *(patchin HF 1.02)* | 31.70 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors) ┊ [![][gh-t8star]](https://huggingface.co/t8star/minimax_h3_ref2va_patchin_hf102/resolve/main/minimax_h3_ref2va_patchin_hf102_T8.safetensors) |
| | ![int8][badge-int8] | ConvRot Lean (HQ) | 21.91 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-INT8-ConvRot-HQ.safetensors) |
| | ![int8][badge-int8] | ConvRot | 20.94 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-INT8-ConvRot.safetensors) |
| | ![int8][badge-int8] | ConvRot Lite | 20.33 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-INT8-ConvRot-Lite.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 (HQ) | 13.60 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-NVFP4-HQ.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 | 10.86 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-NVFP4.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 | 32.05 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4/resolve/main/minimax_h3_ref2va_nvfp4.safetensors) |
| | ![nvfp4][badge-nvfp4] | NVFP4 mixed | 22.76 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_Ref2VA_nvfp4_mixed.safetensors) |
| | ![int4][badge-int4] | NF4 (DiffSynth) | 15.98 GiB | [![][gh-DiffSynth-Studio]](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4/resolve/main/minimax-h3-ref2va-nf4.safetensors) |
| | | OrbitQuant W4A4 | 17.03 GiB | [![][gh-WaveCut]](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/transformer_ref/diffusion_pytorch_model-00001-of-00005.safetensors) |
| | ![nvfp4][badge-nvfp4] | Hybrid NVFP4, FFN-only (Blackwell) | 16.38 GiB | [![][gh-abakanai]](https://huggingface.co/abakanai/Minimax_h3_hybrid/resolve/main/minimax_h3_ref2va_pruned_hybrid_ffn_nvfp4_blackwell.safetensors) |
| | ![nvfp4][badge-nvfp4] | Hybrid NVFP4, QKV+FFN (Blackwell) | 14.03 GiB | [![][gh-abakanai]](https://huggingface.co/abakanai/Minimax_h3_hybrid/resolve/main/minimax_h3_ref2va_pruned_hybrid_nvfp4_blackwell.safetensors) |
| | ![int8][badge-int8] | ⚠️ DT-sQKV ConvRot | 21.00 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-DT-sQKV-INT8-ConvRot.safetensors) |
| | ![int8][badge-int8] | ⚠️ DT-sQKV ConvRot Lean | 27.99 GiB | [![][gh-DmitryDB]](https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV/resolve/main/Ref2VA/MiniMax-H3_Ref2VA-DT-sQKV-INT8-ConvRot-HQ.safetensors) |
| ✓ | ![bf16][badge-bf16] | BF16 | 37.46 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors) |
| ✓ | ![fp8][badge-fp8] | FP8 scaled | 19.52 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors) |
| ✓ | ![int8][badge-int8] | ConvRot | 19.53 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) ┊ [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors) |
| ✓ | ![nvfp4][badge-nvfp4] | NVFP4 | 18.69 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4/resolve/main/minimax_h3_ref2va_pruned_nvfp4.safetensors) |
| ✓ | ![nvfp4][badge-nvfp4] | NVFP4 + ConvRot INT8 | 18.69 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4/resolve/main/minimax_h3_ref2va_pruned_nvfp4_convrot_int8.safetensors) |
| ✓ | ![nvfp4][badge-nvfp4] | NVFP4 | 11.67 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_Ref2VA_pruned_nvfp4.safetensors) |
| ✓ | ![int4][badge-int4] | Mixed INT4/INT8 ConvRot | 14.06 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors) ┊ [![][gh-tsolful]](https://huggingface.co/tsolful/Minimax_H3_INT4MixedConvRot/resolve/main/minimax_h3_ref2va_pruned_INT4BQ.safetensors) |
| ✓ | ![int4][badge-int4] | Mixed INT4/INT8 ConvRot Lean | 17.18 GiB | [![][gh-tsolful]](https://huggingface.co/tsolful/Minimax_H3_INT4MixedConvRot/resolve/main/minimax_h3_ref2va_pruned_INT4Q.safetensors) |
| ✓ | ![int4][badge-int4] | INT4 ConvRot | 15.67 GiB | [![][gh-rockerBOO]](https://huggingface.co/rockerBOO/minimax-h3-nvfp4/resolve/main/minimax_h3_ref2va_pruned_int4_convrot_simple.safetensors) |
| ✓ | ![int4][badge-int4] | W4A8 ConvRot | 11.68 GiB | [![][gh-AX1Y2JP]](https://huggingface.co/AX1Y2JP/MiniMax-H3-W4A8-ConvRot/resolve/main/minimax_h3_ref2va_pruned_symw4a8convrot.safetensors) ┊ [![][gh-Kijai]](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_ref2va_pruned_w4a8_mixed.safetensors) ┊ [![][gh-Winnougan]](https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI/resolve/main/minimax_h3_ref2va_pruned-w4a8_convrot_pruned.safetensors) |

*GGUF quants — see the [GGUF section](#gguf).*

</details>

<details>
<summary><b>Community multi-tier repack</b></summary>

[`DeepBeepMeep/MiniMax-H3`](https://huggingface.co/DeepBeepMeep/MiniMax-H3) collects community files at several precisions. Check the model card for compatibility and licensing.

</details>

<a id="gguf"></a>

#### GGUF Quantized Models

GGUF files are community conversions, not part of the official MiniMax H3 release. Use a compatible loader and follow its documentation.

<details>
<summary><b>Community FL2VA GGUF conversions</b></summary>

| Pruned | Quant | Size | Download |
| :---: | :---: | :---: | :--- |
| | ![Q2_K][badge-Q2_K] | 17.42 GiB † | [![][gh-realrebelai]](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/MiniMax-H3-FL2VA-Q2_K-(Mixed_Precision).gguf) |
| | ![Q3_K_M][badge-Q3_K_M] | 14.50 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q3_K_M.gguf) ┊ [![][gh-realrebelai]](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/MiniMax-H3-FL2VA-Q3_K_M.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q3_K_M.gguf) |
| | ![Q3_K_S][badge-Q3_K_S] | 14.50 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q3_K_S.gguf) |
| | ![Q4_0][badge-Q4_0] | 17.36 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q4_0.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q4_0.gguf) |
| | ![Q4_1][badge-Q4_1] | 20.41 GiB | [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q4_1.gguf) |
| | ![Q4_K_M][badge-Q4_K_M] | 18.50 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q4_K_M.gguf) ┊ [![][gh-realrebelai]](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/MiniMax-H3-FL2VA-Q4_K_M.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q4_K_M.gguf) |
| | ![Q4_K_S][badge-Q4_K_S] | 18.49 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q4_K_S.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q4_K_S.gguf) |
| | ![Q5_0][badge-Q5_0] | 21.21 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q5_0.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q5_0.gguf) |
| | ![Q5_1][badge-Q5_1] | 24.17 GiB | [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q5_1.gguf) |
| | ![Q5_K_M][badge-Q5_K_M] | 22.25 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q5_K_M.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q5_K_M.gguf) |
| | ![Q5_K_S][badge-Q5_K_S] | 22.25 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q5_K_S.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q5_K_S.gguf) |
| | ![Q6_K][badge-Q6_K] | 26.28 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q6_K.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q6_K.gguf) |
| | ![Q8_0][badge-Q8_0] | 33.56 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-FL2VA-Q8_0.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/fl2va/minimax_h3_fl2va-Q8_0.gguf) |
| ✓ | ![IQ1_S][badge-IQ1_S] | **3.78 GiB** — smallest DiT published | [![][gh-MarxistLeninist]](https://huggingface.co/MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF/resolve/main/minimax_h3_fl2va_pruned-IQ1_S.gguf) |
| ✓ | ![IQ1_M][badge-IQ1_M] | 4.22 GiB | [![][gh-MarxistLeninist]](https://huggingface.co/MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF/resolve/main/minimax_h3_fl2va_pruned-IQ1_M.gguf) |
| ✓ | ![Q2_K][badge-Q2_K] | 6.26 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q2_K.gguf) ┊ [![][gh-leejet]](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| ✓ | ![UD-Q2_K_XL][badge-UD-Q2_K_XL] | 7.51 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf) |
| ✓ | ![Q3_K_M][badge-Q3_K_M] | 8.16 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q3_K.gguf) |
| ✓ | ![Q3_K_M][badge-Q3_K_M] | 8.29 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) |
| ✓ | ![UD-Q3_K_XL][badge-UD-Q3_K_XL] | 8.90 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf) |
| ✓ | ![Q4_K_M][badge-Q4_K_M] | 10.64 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q4_K.gguf) ┊ [![][gh-leejet]](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| ✓ | ![Q5_0][badge-Q5_0] | 12.97 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q5_0.gguf) |
| ✓ | ![Q6_K][badge-Q6_K] | 15.45 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q6_K.gguf) |
| ✓ | ![Q8_0][badge-Q8_0] | 19.97 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q8_0.gguf) |
| ✓ | ![Q8_0][badge-Q8_0] | 20.10 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) |

† `realrebelai` ships Q2_K as a **mixed-precision** build, placing it *above* Q3_K_M. When comparing by size, prioritize the numerical value over the quant name.

</details>

<details>
<summary><b>Community Ref2VA GGUF conversions</b></summary>

| Pruned | Quant | Size | Download |
| :---: | :---: | :---: | :--- |
| | ![Q3_K_M][badge-Q3_K_M] | 14.50 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q3_K_M.gguf) ┊ [![][gh-realrebelai]](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/MiniMax-H3-REF2VA-Q3_K_M.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q3_K_M.gguf) |
| | ![Q3_K_S][badge-Q3_K_S] | 14.50 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q3_K_S.gguf) |
| | ![Q4_0][badge-Q4_0] | 17.36 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q4_0.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q4_0.gguf) |
| | ![Q4_1][badge-Q4_1] | 20.41 GiB | [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q4_1.gguf) |
| | ![Q4_K_M][badge-Q4_K_M] | 18.49 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q4_K_M.gguf) ┊ [![][gh-realrebelai]](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/MiniMax-H3-REF2VA-Q4_K_M.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q4_K_M.gguf) |
| | ![Q4_K_S][badge-Q4_K_S] | 18.49 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q4_K_S.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q4_K_S.gguf) |
| | ![Q5_0][badge-Q5_0] | 21.21 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q5_0.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q5_0.gguf) |
| | ![Q5_1][badge-Q5_1] | 24.17 GiB | [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q5_1.gguf) |
| | ![Q5_K_M][badge-Q5_K_M] | 22.25 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q5_K_M.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q5_K_M.gguf) |
| | ![Q5_K_S][badge-Q5_K_S] | 22.25 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q5_K_S.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q5_K_S.gguf) |
| | ![Q6_K][badge-Q6_K] | 26.28 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q6_K.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q6_K.gguf) |
| | ![Q8_0][badge-Q8_0] | 33.56 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/unet/MiniMax-H3-Ref2VA-Q8_0.gguf) ┊ [![][gh-vantagewithai]](https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF/resolve/main/ref2va/minimax_h3_ref2va-Q8_0.gguf) |
| ✓ | ![Q2_K][badge-Q2_K] | 6.22 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q2_K.gguf) |
| ✓ | ![Q3_K_M][badge-Q3_K_M] | 8.12 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q3_K.gguf) |
| ✓ | ![Q4_0][badge-Q4_0] | 10.60 GiB | [![][gh-molbal]](https://huggingface.co/molbal/MiniMax-H3-GGUF) |
| ✓ | ![Q4_K_M][badge-Q4_K_M] | 10.60 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q4_K.gguf) |
| ✓ | ![Q5_0][badge-Q5_0] | 12.94 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q5_0.gguf) |
| ✓ | ![Q6_K][badge-Q6_K] | 14.00 GiB | [![][gh-molbal]](https://huggingface.co/molbal/MiniMax-H3-GGUF) ‡ |
| ✓ | ![Q6_K][badge-Q6_K] | 15.42 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q6_K.gguf) |
| ✓ | ![Q8_0][badge-Q8_0] | 18.77 GiB | [![][gh-molbal]](https://huggingface.co/molbal/MiniMax-H3-GGUF) § |
| ✓ | ![Q8_0][badge-Q8_0] | 19.94 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q8_0.gguf) ┊ [![][gh-molbal]](https://huggingface.co/molbal/MiniMax-H3-GGUF) |

‡ molbal calls this one **`U16G`** — a hand-tuned mixed layout rather than a stock `Q6_K`. § molbal's **`Q8_CR`** is `Q8_0` with ConvRot applied. Both are in the same repo as the plain quants; read the filename.

</details>

<details>
<summary><b>Fine-tuned checkpoint quants</b></summary>

`DmitryDB` also publishes stock-compatible quants of community **fine-tunes** of H3 — fine-tuned QKV weights in blocks 0–31 preserved alongside a tested quantization layout, no custom node or core patch required (the ConvRot / NVFP4 tiers match the base-model tiers exactly: 21.91 / 20.94 / 13.60 / 10.86 GiB, plus a ⚠️ DT-sQKV build at 21.00 GiB).

The fine-tunes themselves are third-party and are not itemised here; browse [`DmitryDB`'s model list](https://huggingface.co/DmitryDB) directly for the current set.

</details>

#### Notes

* **`DmitryDB/MiniMax-H3-INT8-Lean-ConvRot`** and **`DmitryDB/MiniMax-H3-ComfyUI-Quants`** are the *same repo* — the author merged and rebranded. Likewise **`…-INT8-Lean-ConvRot-Dynamic-Time-Separate-QKV`** and **`…-DynTime-sQKV`**. Both names in each pair resolve to the same files, so don't download twice.
* **`t8star/minimax_h3_ref2va_patchin_hf102`** is a weight *modification*, not a quant: +2 % on the 2×2 spatial high-frequency patch in the video-input projection. The author's own tests showed a weak HF-agent gain and did **not** confirm the "oily/waxy" look was removed. Treat as experimental.
* **`Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI`** ships a matching quantized text encoder: [`qwen3vl_32b_minimax_h3-w4a8_convrot.safetensors`](https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI/resolve/main/qwen3vl_32b_minimax_h3-w4a8_convrot.safetensors).
* **`unsloth/MiniMax-H3-GGUF`** also carries Qwen3-VL text-encoder GGUFs: `Q2_K_M` 12.2 GiB and `Q4_K_M` 17.0 GiB.
* **`DmitryDB/MiniMax-H3-ComfyUI-Quants`** also carries VAE files: video VAE FP16 4.85 GiB, audio VAE FP32 577 MiB.
* **`DiffSynth-Studio/MiniMax-H3-NF4`** bundles NF4 TE + video VAE + audio VAE. Requires [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio); the project states a **minimum of 8 GB VRAM** on this path.
* **`WaveCut/MiniMax-H3-OrbitQuant-W4A4`** bundles a quantized TE and FP32 VAE copies, and requires the [`ComfyUI-OrbitQuant`](https://github.com/iamwavecut/ComfyUI-OrbitQuant/tree/feature/minimax-h3-comfyui) node — the W4A4 path is not loadable without it. [Workflow JSON](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-T2VA.json).

<a id="text-encoder"></a>

### Text encoders

MiniMax-H3 uses **Qwen3-VL-32B** for text and vision. On a 24 GB card, you will usually need to reduce the text encoder after reducing the DiT. The options are listed below.

#### Comfy-Org (official repackage)

| Model | Precision | Size | Download |
| :--- | :---: | :---: | :--- |
| `qwen3vl_32b_minimax_h3` | ![bf16][badge-bf16] | 47.97 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors) |
| `qwen3vl_32b_minimax_h3` | ![int8][badge-int8] | 25.28 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) |
| `qwen3vl_32b_minimax_h3` | ![nvfp4][badge-nvfp4] | **14.61 GiB** | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |

The `nvfp4_awq` build at **14.61 GiB** is the smallest official TE and the one to pair with a pruned INT8 DiT on a 24 GB card.

#### Community quantizations

| Model | Precision | Size | Source |
| :--- | :---: | :---: | :--- |
| `qwen3vl_32b_minimax_h3` | ![Q4_K_M][badge-Q4_K_M] | 13.58 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf) |
| `qwen3vl_32b_minimax_h3` | ![int4][badge-int4] | 13.93 GiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int4_convrot.safetensors) |
| `qwen3vl_32b_minimax_h3` | ![nvfp4][badge-nvfp4] | 25.28 GiB † | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| `qwen3vl_32b_minimax_h3` | ![int4][badge-int4] | 15.35 GiB | [![][gh-AX1Y2JP]](https://huggingface.co/AX1Y2JP/MiniMax-H3-W4A8-ConvRot) — W4A8 ConvRot |
| `qwen3vl_32b_minimax_h3` | ![int4][badge-int4] | — | [![][gh-Winnougan]](https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI/resolve/main/qwen3vl_32b_minimax_h3-w4a8_convrot.safetensors) — W4A8 ConvRot |
| `qwen3vl_32b_minimax_h3` | ![Q2_K][badge-Q2_K] | 12.2 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF) — `Q2_K_M` GGUF |
| `qwen3vl_32b_minimax_h3` | ![Q4_K_M][badge-Q4_K_M] | 17.0 GiB | [![][gh-unsloth]](https://huggingface.co/unsloth/MiniMax-H3-GGUF) — `Q4_K_M` GGUF |
| `qwen3vl_32b_minimax_h3` | ![Q2_K][badge-Q2_K] | **7.91 GiB** | [![][gh-realrebelai]](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) — smallest TE published |
| `qwen3vl_32b_minimax_h3` | ![int8][badge-int8] | 24.89 GiB | [![][gh-DeepBeepMeep]](https://huggingface.co/DeepBeepMeep/MiniMax-H3) — quanto-INT8 ⚠️ no license |

† `Abiray`'s `nvfp4_awq` file is byte-for-byte the size of Comfy-Org's **INT8** build, not of an NVFP4 one. Check the file before assuming it is a smaller download.

<a id="components"></a>

### VAE and components

<a id="components-vae"></a>

#### VAE (video & audio)

Both VAEs are **required** for every generation workflow — H3 decodes video and audio through separate autoencoders.

| Component | Source | Precision | Size | Download |
| :--- | :--- | :---: | :---: | :--- |
| Video VAE | Comfy-Org | ![fp16][badge-fp16] | 4.85 GiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) |
| Audio VAE | Comfy-Org | ![fp32][badge-fp32] | 577 MiB | [![][gh-Comfy--Org]](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) |
| Video VAE | dummy9996 | ![fp8][badge-fp8] | 2.60 GiB | [![][gh-dummy9996]](https://huggingface.co/dummy9996/minimax_h3_vae_fp8/resolve/main/minimax_h3_video_vae_fp8mix.safetensors) |
| Audio VAE | dummy9996 | ![bf16][badge-bf16] | 289 MiB | [![][gh-dummy9996]](https://huggingface.co/dummy9996/minimax_h3_vae_fp8/resolve/main/minimax_h3_audio_vae_bf16.safetensors) |
| Video VAE | Kijai | ![int8][badge-int8] | 2.95 GiB | [![][gh-Kijai]](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_video_vae_int8_convrot.safetensors) |

The `fp8mix` video VAE (**2.60 GiB**) plus the `bf16` audio VAE (**289 MiB**) save roughly 2.5 GiB over the official pair — worth taking on a 12–16 GB card, where the VAE competes with the DiT for the same headroom.

<a id="tae"></a>

#### Tiny Autoencoder (TAE) — previews only

A quickly-trained 2D tiny VAE by [Kijai](https://huggingface.co/Kijai/MiniMax-H3-TAE). The author's own assessment: not a great outcome, but it still beats `latent2rgb` for previews. **9 MiB.** Currently only usable through the `ModelPreviewOverride` node in [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes).

| Component | Size | Download |
| :--- | :---: | :--- |
| TAE (preview VAE) | 9 MiB | [![][gh-Kijai]](https://huggingface.co/Kijai/MiniMax-H3-TAE/resolve/main/vae_approx/taeh3.safetensors) |

<a id="imagevae"></a>

#### Image VAE (single-frame)

An experimental image-specialised H3 VAE that decodes a single temporal latent (`T=1`) into one still. Merged checkpoint — no custom node needed.

⚠️ **For image workflows only.** The image-tuned decoder materially regresses multi-frame video reconstruction, so keep the original VAE loaded for video.

| Component | Size | Download |
| :--- | :---: | :--- |
| Single-image VAE (step 1597) | 4.85 GiB | [![][gh-Mamad8]](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE/resolve/main/minimax_h3_t1_image_vae_step1597.safetensors) |

<a id="refpatch"></a>

#### Ref Patch — FL2VA that behaves more like Ref2VA

Diffs the **112 keys shared** between the `ref2va` and `fl2va` weights and stores the differences as a single 148 MiB patch, letting the lighter FL2VA checkpoint partially mimic Ref2VA behaviour. Apache-2.0. Requires the [`ComfyUI-MiniMaxH3_Ref-Patch`](https://github.com/lihaoyun6/ComfyUI-MiniMaxH3_Ref-Patch) node.

| Component | Size | Download |
| :--- | :---: | :--- |
| Ref Patch | 148 MiB | [![][gh-lihaoyun6]](https://huggingface.co/lihaoyun6/MiniMax-H3-Ref-Patch) |

<a id="engines"></a>

### Runtimes

| Engine | ⭐ | H3 support |
| :--- | ---: | :--- |
| [`ComfyUI`](https://github.com/comfyanonymous/ComfyUI) | 127159 | Native, day-0. INT8 is now in mainline (commit `1a510f04`) — see [Compatibility](#compat) before reusing older INT8 quants. |
| [`modelscope/DiffSynth-Studio`](https://github.com/modelscope/DiffSynth-Studio) | 12925 | `MiniMaxH3Pipeline` in `diffsynth.pipelines.minimax_h3_audio_video`; docs at `docs/en/Model_Details/MiniMax-H3.md`, examples at `examples/minimax_h3/`. Ships **NF4** quantized inference with an **8 GB VRAM** floor. |
| [`ModelTC/LightX2V`](https://github.com/ModelTC/LightX2V) | 2655 | Full inference support: parallelism, quantized DiT, feature caching. Scripts at `scripts/minimax_h3`. Also the home of the Turbo 4-step / 768p LoRAs. |
| [`NVlabs/Sana` Sol-Engine](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3) | 8812 | Hardware-specific full-stack runtimes (kernel optimization, Sol-Attn, approximate caching, memory-efficient decoding) on the `sol-engine` branch, **Apache-2.0**. Released full-opt configs for GB200, GB10, H100, A100, RTX 4090 and RTX 5090 with **3.55–4.52x** speedup vs the matching baseline on the same hardware (e.g. 4.44x on a single RTX 4090, 768p@5s); launch via `scripts/launch_config.py`, local or Slurm. |
| [`MiniMax-AI/MiniMax-H3`](https://github.com/MiniMax-AI/MiniMax-H3) | 5536 | The official repository — reference implementation and prompt guides. |
| [`antirez/h3.c`](https://github.com/antirez/h3.c) | 1652 | Apple Silicon native Metal engine, **MIT**, tutorial in the README. T2V/A, first-last-frame, and ordered Ref2VA references all working. |
| [`MiniMaxH3ComfyUI/MiniMax-H3-ComfyUI`](https://github.com/MiniMaxH3ComfyUI/MiniMax-H3-ComfyUI) | 101 | Runs the 33B + Turbo LoRA locally with SGLang / vLLM / diffusers as selectable backends; T2V / I2V / R2V templates included. |
| [`unslothai/unsloth`](https://github.com/unslothai/unsloth) | 70709 | Lists MiniMax-H3 among the models it can run and train. |

<a id="nodes"></a>
<a id="recipes"></a>

## ComfyUI workflows & nodes

### Conditioning & orchestration

| Node | ⭐ | What it does |
| :--- | ---: | :--- |
| [`comfyui-minimax-h3-audio-T8`](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) ![Conditioning][cat-cond] | 653 | v1.17.0, **62 nodes** across eight menus: Audio (stable), Audio Experimental (multi-rate), Still, Conditioning, Models, Long Video, Speech, Source AV. Baseline ComfyUI `0.31.0`, commit `cbbc9dab1`, Python 3.10+. |
| [`ComfyUI_MiniMaxH3_Director`](https://github.com/huangserva/ComfyUI_MiniMaxH3_Director) ![Conditioning][cat-cond] | 553 | Provides five importable JSON templates: t2v, fl2v, r2v, v2v, and rv2v. |
| [`ComfyUI-H3-Motion-Context`](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) ![Conditioning][cat-cond] | 491 | Chains clips to maintain motion and sound continuity: clip A's last frames and audio inform clip B's start. **Patches at runtime only**; re-validates against current ComfyUI source on each start, refusing to run on mismatch — the safest patching approach in this list. |
| [`ComfyUI_MiniMaxH3_Director`](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) ![Conditioning][cat-cond] | 359 | The original Director. |
| [`ComfyUI-MiniMaxH3-Easy`](https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy) ![Conditioning][cat-cond] | 332 | Provides a compact workflow for T2V, I2V, first/last-frame, and reference video. Supports unified multi-media input with `@` references and inline dialogue blocks. |
| [`ComfyUI-MiniMaxH3-Director`](https://github.com/seesee75-commits/ComfyUI-MiniMaxH3-Director) ![Conditioning][cat-cond] | 182 | Offers a timeline editor: drag media onto tracks, trim on a ruler, assign one prompt per shot, with live sampling preview, retakes, and shot chaining. The compiled final prompt remains visible during editing. |
| [`ComfyUI-PainterNodes`](https://github.com/princepainter/ComfyUI-PainterNodes) ![Conditioning][cat-cond] | 178 | `MiniMaxRefToVideo2` node supports the official reference and dialogue format. |
| [`OpenH3-IR`](https://github.com/ruashots/open-h3-ir) ![Conditioning][cat-cond] | 19 | The ComfyUI side of OpenH3-IR (see [Prompting](#recipes-prompt)), using its Context-IR service from the same repo, on a model you already run. Drop your pictures, clips and sounds on one panel, name them, then mention them by name in the sentence. The nodes and the service are one tool listed twice, not two projects. |

### Upscaling, loading & repair

| Node | ⭐ | What it does |
| :--- | ---: | :--- |
| [`scraed/LanPaint`](https://github.com/scraed/LanPaint) ![Conditioning][cat-cond] | 1331 | Performs training-free video and audio inpainting; H3 support was fixed in v2.1.0. |
| [`ComfyUI-MiniMaxH3_LatentUpscaler`](https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler) ![Upscaling][cat-upscale] | 191 | Latent spatial upscaler for H3's `NestedTensor` AV latents (video `[B,24,T,H/16,W/16]` + audio `[B,32,2,T_audio]`), which stock `LatentUpscaleBy` cannot process. Re-noises video and audio for two-pass sampling and scales `minimax_refs` / `minimax_keyframes` conditioning. `audio_denoise`: **0** locks audio, **1** fully remixes, **0.25–0.5** for light remixing. |
| [`Comfyui_Minimax_h3_latent_Upscaler`](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) ![Upscaling][cat-upscale] | 151 | **Learned** neural upscaler for H3's 24-channel latents — generate at low resolution, upscale the latent in place, then refine at target resolution, skipping the `decode → pixel upscale → encode` round-trip through the ~5B VAE and the ghosting that naive latent interpolation introduces. Continuous **1.0–4.0×** scale. [Checkpoints on HF](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) (bf16/fp16 ~691 MB, fp32 ~1.38 GB, Apache-2.0) with I2V and R2V example workflows; trained on ~80k paired samples. |
| [`ComfyUI-INT8-Fast`](https://github.com/BobJohnson24/ComfyUI-INT8-Fast) ![Acceleration][cat-accel] | 286 | **Largely superseded** as INT8 is now native in ComfyUI. Its remaining value is `convert_comfy_quant.py`; see [Compatibility](#compat). |

<a id="wf"></a>
<a id="wf-comfyui"></a>

### Templates & example workflows

Official ComfyUI templates (these ship with ComfyUI; links allow viewing the graph without launching the app):

* [Text-to-Video (T2V)](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json) · [Image-to-Video (I2V)](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json) · [Reference-to-Video (R2V)](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json)

Community workflows:

* [MiniMax-H3 FL2V GGUF workflow](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/minimax_fl2v_gguf_workflow.json) — Loads and runs the GGUF-quantized FL2VA model.
* [`joeygambino/MiniMax-H3-Multishot-Workflow`](https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow) — Strings several FL2VA/Ref2VA clips into one continuous sequence with matched audio handoffs. Apache-2.0 licensed.
* [`javawock7618/comfy-MiniMax-H3-workflows`](https://huggingface.co/javawock7618/comfy-MiniMax-H3-workflows) — Bundles the entire low-VRAM acceleration stack into one importable workflow.
* [OrbitQuant T2VA](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-T2VA.json) · [T2VA API form](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-T2VA-api.json) · [Ref2VA API form](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-Ref2VA-api.json) — require [`ComfyUI-OrbitQuant`](https://github.com/WaveCut/ComfyUI-OrbitQuant).

<a id="recipes-prompt"></a>

## Prompting

H3 prompts have a fixed three-part structure, inline `<Picture X>` / `<Video X>` / `<Audio X>` reference tags, and `<d>` for dialogue. Start with official guides, then use one prompt tool at a time. For reference audio, a clean, clearly spoken 10-second clip is picked up more reliably than a noisy one.

**Read first:** [Base prompt guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/VIDEO_PROMPT_WRITING_GUIDE.md) · [Reference-mode prompt guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/VIDEO_PROMPT_WRITING_GUIDE_REF.md)

| Tool | Why you'd pick it |
| :--- | :--- |
| [`ComfyUI-MiniMax-H3-Promptor`](https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor) | From v1.1.0, embeds `<Picture X>` directly into the narrative action line for "zero-hallucination inline annotation." Decouples visual analysis from text structuring, reducing API cost. |
| [`ComfyUI-MiniMax-H3-Guide`](https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide) | Zero dependencies. "Typed Plan v2" splits identity, keyframes, motion, edit source, voice, and score into explicit roles, compiles them into valid H3 prose, and routes to native nodes. Includes reusable image/audio reference sheets and a locked-frame Foley mode. |
| [`OpenH3-IR`](https://github.com/ruashots/open-h3-ir) | Follows the Context-IR format and examples MiniMax published, rather than rewriting your prompt: it writes the document, then checks it and fixes what is wrong. Command line, HTTP, or its own ComfyUI nodes, on a model you already run. Apache-2.0. |
| [`comfyui-minimax-h3-prompt-enhancer-T8`](https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8) | Provides server-side prompt enhancement via `doubao-seed-evolving`. |
| [`awesome-minimax-h3-prompts`](https://github.com/BeatAPI/awesome-minimax-h3-prompts) | A prompt corpus with WebM examples and author attribution, categorized into story, action/fantasy, ad/product, music performance, and vlog. |
| [`minimax-h3-prompt-skill-T8`](https://github.com/T8mars/minimax-h3-prompt-skill-T8) | "Creative DNA" case library, installable as an agent skill, with an Electron desktop viewer. |

<a id="agent-skills"></a>

### Agent skills

Skill packages that let a coding agent drive H3 without the ComfyUI canvas. They differ in how far they take you, and are listed here from the most complete: one carries a brief through to a finished file, one drives a local ComfyUI, and the rest write the prompt and hand it back for you to run.

| Skill repo | Runs against | What it does |
| :--- | :--- | :--- |
| [`awesome-minimax-h3`](https://github.com/joeVenner/awesome-minimax-h3) | Hosted API | Carries a brief through to a finished file against the hosted MiniMax API: submit, poll with backoff, resolve the returned `file_id`, then post-process with `ffmpeg`. Reports that the documented `/v2/…/query` polling form returns 404 and gives `GET /v1/query/video_generation` as the working call. Three `SKILL.md` packages — `minimax-video` for H3, plus Speech 2.8 and Music 3.0 skills that compose with it. Needs a `MINIMAX_API_KEY`, not local weights. MIT. |
| [`ComfyUI-Agent-Kit`](https://github.com/SlavaSexton/ComfyUI-Agent-Kit) | Local ComfyUI | Drives a local ComfyUI end to end from Claude Code, Codex, Gemini CLI, or Qwen Code — workflow building, hardware-aware model selection, multi-shot video. Ships a standalone H3 skill among its 581 templates. Apache-2.0. |
| [`Minimax-H3-Prompt-AgentSkill`](https://github.com/benjiyaya/Minimax-H3-Prompt-AgentSkill) | Prompt only | Builds a formatted H3 prompt from your media and idea, then hands it back for you to run. No license stated. |
| [`minimax-h3-opencode-skills`](https://github.com/unknowlei/minimax-h3-opencode-skills) | Prompt only | OpenCode skill suite covering directing, routing, multi-shot planning, prompt generation, and review. MIT. |

Also relevant: [`ComfyUI-PainterNodes`](https://github.com/princepainter/ComfyUI-PainterNodes) (`MiniMaxRefToVideo2`, with the official reference and dialogue format), and [`minimax-h3-prompt-skill-T8`](https://github.com/T8mars/minimax-h3-prompt-skill-T8) in the table above, which ships its case library as an installable skill.

<a id="speed"></a>

## Speed

Two levers stack: Turbo LoRAs cut step count from ~20 to 4–8, and caching or kernel work makes each step cheaper. Check your PyTorch build first; an outdated CUDA build commonly causes slow generations.

<a id="turbo"></a>

### Turbo (Acceleration LoRA)

Turbo LoRAs are community acceleration models. Start with [`ModelTC/Minimax-H3-Turbo`](https://github.com/ModelTC/Minimax-H3-Turbo) and [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo), then use the workflow instructions provided by the project you choose. At 4 steps the audio track can degrade along with fast motion; 6–8 steps helps both.

<details>
<summary><b>Community Turbo checkpoint reference</b></summary>

| Variant | Steps | Base | Precision | Size | Download |
| :--- | :---: | :---: | :--- | :---: | :--- |
| `fl2v v0.1` | 4 | Full | ![bf16][badge-bf16] | 1.29 GiB | [![][gh-lightx2v]](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v0.1.safetensors) |
| `fl2v v1.0 768p` | 4 | Full | ![bf16][badge-bf16] | 1.29 GiB | [![][gh-lightx2v]](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors) |
| `fl2v v1.0 768p · comfyui` | 4 | Full | ![bf16][badge-bf16] | 1.82 GiB | [![][gh-lightx2v]](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors) |
| `fl2v v1.0` | 8 | Full | ![bf16][badge-bf16] | 1.29 GiB | [![][gh-lightx2v]](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors) |
| `fl2v v1.0 · comfyui` | 8 | Full | ![bf16][badge-bf16] | 1.82 GiB | [![][gh-lightx2v]](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) |
| `lightx2v v0.1` | 4 | Full | ![bf16][badge-bf16] | 1.82 GiB | [![][gh-Kijai]](https://huggingface.co/Kijai/MiniMax-H3_comfy/resolve/main/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors) |
| `lightx2v v0.1 · rank-21 resize` | 4 | Full | ![bf16][badge-bf16] | 300 MiB | [![][gh-Kijai]](https://huggingface.co/Kijai/MiniMax-H3_comfy/resolve/main/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors) |
| `fl2v` | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step.safetensors) |
| `fl2v ema` | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ema.safetensors) |
| `fl2v ckpt500` | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ckpt500.safetensors) |
| `fl2v ema ckpt500` | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ema_ckpt500.safetensors) |
| `fl2v ckpt850` ← best 4-step under motion | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ckpt850.safetensors) |
| `fl2v ema ckpt850` | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ema_ckpt850.safetensors) |
| `fl2v v4 step600` | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_v4_step600.safetensors) |
| **`fl2v v4 step600 ema`** ← recommended default | 4 | Full | ![bf16][badge-bf16] | 744 MiB | [![][gh-larryvrh]](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_v4_step600_ema.safetensors) |
| `fl2v pruned` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_4step_pruned_comfyui.safetensors) |
| `fl2v pruned ema` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_4step_ema_pruned_comfyui.safetensors) |
| `fl2v pruned ckpt500` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors) |
| `fl2v pruned ema ckpt500` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_4step_ema_ckpt500_pruned_comfyui.safetensors) |
| `fl2v pruned ckpt850` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors) |
| `fl2v pruned ema ckpt850` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_4step_ema_ckpt850_pruned_comfyui.safetensors) |
| `fl2v pruned v4 step600` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors) |
| `fl2v pruned v4 step600 ema` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors) |
| `fl2v v1.0 768p · rank-21 resize` | 4 | Pruned | ![bf16][badge-bf16] | 298 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors) |
| `fl2v v1.0 · rank-21 resize` | 8 | Pruned | ![bf16][badge-bf16] | 327 MiB | [![][gh-drbaph]](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_resized_avg_rank_21_bf16.safetensors) |
| `fl2v pruned ckpt500 V1` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt500_V1.safetensors) |
| `fl2v pruned ckpt600 V4` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt600_V4.safetensors) |
| `fl2v pruned ckpt600 ema V4` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt600_ema_V4.safetensors) |
| `fl2v pruned ckpt850 V1` | 4 | Pruned | ![bf16][badge-bf16] | 592 MiB | [![][gh-Abiray]](https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt850_V1.safetensors) |
| `fl2v diffusers` | 4 | Full | ![bf16][badge-bf16] | 0.79 GiB | [![][gh-InstantX]](https://huggingface.co/InstantX/MiniMax-H3-Turbo-Lora-Diffusers) |
| `fl2v` | 4 | Full | ![bf16][badge-bf16] | 717 MiB | [![][gh-joyfox]](https://huggingface.co/joyfox/MiniMax-H3-Turbo/resolve/main/minimax_h3_fl2va_4step_lora.safetensors) |
| `fl2v step 100` | 8 NFE | Full | ![bf16][badge-bf16] | 738 MiB | [![][gh-tutututututu]](https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA/resolve/main/comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-bf16-comfyui.safetensors) |
| `fl2v step 200` | 8 NFE | Full | ![bf16][badge-bf16] | 738 MiB | [![][gh-tutututututu]](https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA/resolve/main/comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000200-bf16-comfyui.safetensors) |
| `fl2v step 300` | 8 NFE | Full | ![bf16][badge-bf16] | 738 MiB | [![][gh-tutututututu]](https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA/resolve/main/comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000300-bf16-comfyui.safetensors) |
| `fl2v 4-step` · ConvRot · ⚠️ dual-clock sampler or 8–10 steps | 4 | Full | ![int8][badge-int8] | 779.9 MiB | [![][gh-t8star]](https://huggingface.co/t8star/minimax-h3-4step-turbo-loras-comfyui-exp) |
| `fl2v 4-step ema` · ConvRot | 4 | Full | ![int8][badge-int8] | 779.9 MiB | [![][gh-t8star]](https://huggingface.co/t8star/minimax-h3-4step-turbo-loras-comfyui-exp) |
| `fl2v v4 step600 (T8-convert)` · ConvRot | 4 | Full | ![int8][badge-int8] | 779.9 MiB | [![][gh-t8star]](https://huggingface.co/t8star/minimax-h3-4step-turbo-loras-comfyui-exp/resolve/main/minimax_h3_turbo_v4_step600_comfyui_T8-convert.safetensors) |
| `lightx2v v0.1 · alpha8 T8-convert` · ConvRot · ⚠️ dual-clock sampler or 8–10 steps | 4 | Full | ![int8][badge-int8] | 1.82 GiB | [![][gh-t8star]](https://huggingface.co/t8star/minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert) |
| `fl2v v1.0 768p` · ConvRot · needs ComfyUI-LoraInt8Loader | 4 | Full | ![int8][badge-int8] | 991 MiB | [![][gh-rzgar]](https://huggingface.co/rzgar/minimax_h3_fl2v_lightx2v_4step_int8-convrot_comfy/resolve/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16_int8convrot.safetensors) |
| `fl2v v1.0` · ConvRot · needs ComfyUI-LoraInt8Loader | 8 | Full | ![int8][badge-int8] | 991 MiB | [![][gh-rzgar]](https://huggingface.co/rzgar/minimax_h3_fl2v_lightx2v_4step_int8-convrot_comfy/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16_int8convrot.safetensors) |
| `lightx2v v0.1` · ConvRot · needs ComfyUI-LoraInt8Loader | 4 | Full | ![int8][badge-int8] | 991 MiB | [![][gh-rzgar]](https://huggingface.co/rzgar/minimax_h3_fl2v_lightx2v_4step_int8-convrot_comfy/resolve/main/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_int8convrot.safetensors) |
| `fl2v CMF` | 4 | Full | Q4TP (CMF) | 25.20 GiB | [![][gh-infosave]](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/mmh3-turbo-q4tp.cmf) |
| `fl2v CMF · FL2VA` | 4 | Full | Q4TP (CMF) | 25.70 GiB | [![][gh-infosave]](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/mmh3-turbo-fl2va-q4tp.cmf) |
| `fl2v CMF · FL2VA (smaller)` | 4 | Full | Q2TP (CMF) | 20.12 GiB | [![][gh-infosave]](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/mmh3-turbo-fl2va-q2tp.cmf) |

</details>

### Acceleration nodes

The figures below come from each project's own testing.

| Node | ⭐ | Mechanism & published parameters |
| :--- | ---: | :--- |
| [`ComfyUI-Spectrum-MiniMax-H3`](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) ![Acceleration][cat-accel] | 493 | Spectral feature forecasting — fits post-transformer features with **Chebyshev ridge regression** and extrapolates future steps, skipping selected transformer evaluations. Adaptive scheduling with native fallbacks. The author is explicit that this is an approximation: **output is not bit-identical to native.** |
| [`ComfyUI-SolAttn_triton`](https://github.com/kijai/ComfyUI-SolAttn_triton) ![Acceleration][cat-accel] | 266 | SolAttention Triton kernel — optimized attention for H3 and other Sol-Attn models. |
| [`TE-Speed-MiniMaxH3`](https://github.com/tl2012tl/TE-Speed-MiniMaxH3) ![Acceleration][cat-accel] | 2 | Block-cache accelerator, by the original TE-Speed author. v3.2 targets current ComfyUI (block prefetch, no core patch), adds a 4/8-step LoRA mode with automatic strategy selection by step count, a long-video (>10 s) cache strategy, and chunked CPU residual transfer; bundles a TE-Speed-compatible fork of the Sol-Attn node. ⚠️ Ships a compiled `nodes.pyd`, no license stated. |

<a id="training"></a>

## Training & LoRAs

> **Training status.** H3 includes weights and inference code, but no official trainer. The Hugging Face Diffusers integration is also inference-only. Everything below comes from the community.

| Project | ⭐ | Notes |
| :--- | ---: | :--- |
| [`radixark/miles_diffusion`](https://github.com/radixark/miles_diffusion) | 42 | Flow-GRPO RL post-training against reward models. The H3 [t2va recipe](https://miles.radixark.com/docs/diffusion/models/h3/h3) is verified on 2×H200. |
| [`IAmIronMan42/MiniMax-H3-FineTuning`](https://github.com/IAmIronMan42/MiniMax-H3-FineTuning) | 487 | **The most complete trainer currently available.** Supervised rectified-flow training on top of the official Diffusers implementation, with latent caching (`prepare_cache.py`, `prepare_cache_pairs.py`) and a `FIXES.md` documenting nine fixes the author needed to make it converge. Verified scale: LoRA on **8×A800**, 2000 clips of ~30 s at 448×768, ~65k tokens per sequence, **stereo audio inside the loss**. |
| [`shootthesound/Fizgig`](https://github.com/shootthesound/Fizgig) | 157 | LoRA / LoKr training studio with a built-in **"✨ MiniMax H3 Fast"** preset (LoKr, 8 dim / alpha 16, 60 epochs). Also does profile / repair / extract. |
| [`inlineresearch/Inline-Studio`](https://github.com/inlineresearch/Inline-Studio) | 213 | Node-canvas film tool that trains H3 LoRAs on a local GPU. States **"MiniMax H3 (4-bit, video) ~20.6 GB"**. |
| [`ModelTC/LightX2V`](https://github.com/ModelTC/LightX2V) | 2655 | The training side of Turbo distillation. The DMD config is public at `configs/minimax_h3/dmd`. |
| [`unslothai/unsloth`](https://github.com/unslothai/unsloth) | 70709 | Lists MiniMax-H3 among the models it can run and train. Check the current Unsloth documentation for the H3 workflow that fits your setup. |

<a id="lora"></a>

### Ready-made LoRAs

Acceleration LoRAs live in [Turbo](#turbo). This section covers everything else.

### Styles

| LoRA | Size | What it does |
| :--- | :---: | :--- |
| [![][gh-matlod]](https://huggingface.co/matlod/minimax-h3-turnaround) **minimax-h3-turnaround** | 60 MiB each | **Contact-Sheet diffusion** — one reference image + one instruction produces five coherent, progressively rotated views of the same subject in a single pass, by using H3's timeline as a slot axis rather than as time. A character turnaround from one photo: **~10 s at 512², ~57 s at 1024²**. Three builds: `1024-cont/s600`, `512/s1500`, `512-instruct/s400`. |
| [![][gh-fal]](https://huggingface.co/fal/research-mini-max-h3-realism-people-lora) **Realism — People** | 125 MiB | Natural-looking people in everyday scenarios, trained by fal on diverse photo data. Works across T2V / I2V / R2V. |
| [![][gh-Inner--Reflections]](https://huggingface.co/Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime) **Looping Sketch Anime** | 569 MiB | Hand-drawn 2D outlines, flat colours, white outline, built to loop. Strength **0.75–1.25**; pair with a Turbo LoRA if you want to push toward the high end. |
| [![][gh-TenStrip]](https://huggingface.co/TenStrip/Krea2-H3-Style-Lora) **Krea2 Style** | 1.03 GiB (r512) / 266 MiB (r128) | Experimental style shift toward Krea 2's aesthetics, extracted from a cross-model attn-graft (Krea2 attn K/V). **Only effective for pure T2V.** The author states r512 captures 52 % of the extraction target and r128 only 24 % (negligible) — prefer r512. MiniMax-H3 Community License. |

### Utility

| LoRA | Size | What it does |
| :--- | :---: | :--- |
| [![][gh-mvp--lab]](https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA) **RAVEN Streaming (4-NFE preview)** | 4.71 GiB | Turns H3 into a **causal streaming generator** — video is produced chunk by chunk, each extrapolated from prior content instead of denoised as one bidirectional clip (`r=128`, 4 NFE, 192 frames at 768×1376/24 fps, `sink=2` / `window=2`). Preview release by the [RAVEN](https://github.com/mvp-ai-lab/RAVEN) team (Imperial College); the authors state texture detail is still limited and real-time playback needs further inference acceleration. Runs through the RAVEN code repo, not ComfyUI. MiniMax-H3 Community License. |
| [![][gh-lightx2v]](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA) **Prompt Rewriter** | 3.48 GiB | A Qwen3.6-27B fine-tune that rewrites a short prompt into H3's expected three-part structure. This is a *language-model* LoRA — it does not load into the DiT. |
| [![][gh-lightx2v]](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B) **Prompt Rewriter 8B** | 2.60 GiB | The multimodal sibling of the Prompt Rewriter above: a Qwen3-VL-8B-Instruct LoRA that also *sees your keyframes* — covers T2VA, I2VA (first frame), L2VA (last frame), and FL2VA rewriting with an integrated shot timeline and synchronized physical/ambient sound and music guidance. Ref2VA is not supported. Also a *language-model* LoRA — it does not load into the DiT. |

<a id="partners"></a>

## Serving H3

These projects are the main open-source options for serving H3 outside a ComfyUI workflow. SGLang and vLLM-Omni both ship official, hardware-verified H3 recipes; pick by ecosystem and by which published GPU profile matches your hardware.

| Project | Where it fits |
| :--- | :--- |
| [SGLang](https://github.com/sgl-project/sglang) | First-class H3 serving. SGLang Diffusion ships a native H3 pipeline with an official [cookbook](https://docs.sglang.ai/cookbook/diffusion/MiniMax/MiniMax-H3): `sglang serve --model-variant fl2va\|ref2va` covers t2va, fl2va, and ref2va (V2V is a ref2va use case), with every published recipe verified on real hardware — 4×H200/H100, B200/B300 (online quantization), 2×RTX 5090 layerwise offload, single RTX 4090 24 GB (`kitchen_int8`), and day-0 AMD MI355X/MI300X via ROCm + AITER. Also the selectable backend in [`MiniMax-H3-ComfyUI`](https://github.com/MiniMaxH3ComfyUI/MiniMax-H3-ComfyUI). |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni) | First-class H3 serving on the vLLM stack. Since v0.26.0 it ships an official [MiniMax-H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md): OpenAI-compatible `/v1/videos` serving of T2VA, FL2VA, and Ref2VA from one diffusion stage that loads both DiTs and shares the TE and VAEs. Published profiles from 2× RTX 4090/5090 (CPU offload, ≥200 GiB host RAM) up to 4× B300/GB200, plus AMD ROCm. |
| [vLLM](https://github.com/vllm-project/vllm) | The underlying engine vLLM-Omni extends — vLLM alone serves autoregressive models, not H3's DiT. Deploy H3 through vLLM-Omni; keep plain vLLM for the text-model side of your stack (e.g. a prompt-rewriter endpoint). |

<a id="compat"></a>

## Compatibility & Licensing

### ComfyUI INT8

ComfyUI includes native INT8 support. Older I8Fast files use different tensor names; use [`convert_comfy_quant.py`](https://github.com/BobJohnson24/ComfyUI-INT8-Fast) or download a native-format quant.

### ComfyUI patches

Some community tools modify or patch ComfyUI. Check the project's documentation and keep your ComfyUI version compatible.

| Type | Project | Notes |
| :--- | :--- | :--- |
| Runtime patch | [`DmitryDB/MiniMax-H3-DynTime-sQKV`](https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV) | Required for its DT-sQKV files. |
| Runtime patch | [`ComfyUI-H3-Motion-Context`](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) | Checks its ComfyUI assumptions at startup. |

### Reported environments

* [`comfyui-minimax-h3-audio-T8`](https://github.com/T8mars/comfyui-minimax-h3-audio-T8): ComfyUI `0.31.0`, commit `cbbc9dab1`, Python 3.10+.
* [`ComfyUI_MiniMaxH3_Director`](https://github.com/huangserva/ComfyUI_MiniMaxH3_Director): RTX 4090 48 GB, ComfyUI 0.30.0, PyTorch 2.11.0, CUDA 12.8, Ref2VA INT8.

### Licenses

| License | Where |
| :--- | :--- |
| Apache-2.0 | `ModelTC/Minimax-H3-Turbo` and the Turbo LoRA line · Ref Patch |
| MIT | `antirez/h3.c` |
| No license stated | `DeepBeepMeep/MiniMax-H3` |

For other projects, check the repository or model card.

<a id="credits"></a>

## Acknowledgements

This index is only possible because other people made the models, tools, tests, and documentation it points to. Thank you to the MiniMax, ComfyUI, SGLang, vLLM, NVIDIA, and Unsloth teams, and to the independent maintainers who keep testing H3 on real hardware.

The structure and much of the resource discovery come from the community-maintained [`wildminder/awesome-minimax-H3`](https://github.com/wildminder/awesome-minimax-H3). This page follows that work and adds MiniMax's own GitHub and Hugging Face scan.

More specific thanks go to:

* [Comfy-Org](https://huggingface.co/Comfy-Org) for the official ComfyUI conversions, workflow templates, and day-one support.
* [ModelTC / LightX2V](https://github.com/ModelTC/LightX2V) for the Turbo distillation work and public training configuration.
* [`Larryvrh`](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo) for the checkpoint comparisons behind the Turbo guidance.
* [`Kijai`](https://github.com/kijai/ComfyUI-SolAttn_triton) for the NVIDIA Sol-Attn implementation and benchmark notes.
* [`IAmIronMan42`](https://github.com/IAmIronMan42/MiniMax-H3-FineTuning) for the training work and the documented fixes.
* [Salvatore Sanfilippo (antirez)](https://github.com/antirez) for the standalone H3 C/Metal inference engine.
* Every quantizer and workflow maintainer represented above. Their files, testing time, and write-ups make local H3 use much easier.

If you spot a wrong number, broken link, or missing compatibility note, please open an issue or send a correction.

## Contact

For MiniMax H3 questions, contact [model@minimax.io](mailto:model@minimax.io).

<!-- MARKDOWN LINKS & IMAGES -->
[gh-MiniMaxAI]: https://img.shields.io/badge/%F0%9F%A4%97-MiniMaxAI-FFD21E?style=flat-square
[gh-Comfy--Org]: https://img.shields.io/badge/%F0%9F%A4%97-Comfy--Org-FFD21E?style=flat-square
[gh-Abiray]: https://img.shields.io/badge/%F0%9F%A4%97-Abiray-FFD21E?style=flat-square
[gh-DmitryDB]: https://img.shields.io/badge/%F0%9F%A4%97-DmitryDB-FFD21E?style=flat-square
[gh-DiffSynth-Studio]: https://img.shields.io/badge/%F0%9F%A4%97-DiffSynth--Studio-FFD21E?style=flat-square
[gh-DeepBeepMeep]: https://img.shields.io/badge/%F0%9F%A4%97-DeepBeepMeep-FFD21E?style=flat-square
[gh-WaveCut]: https://img.shields.io/badge/%F0%9F%A4%97-WaveCut-FFD21E?style=flat-square
[gh-dummy9996]: https://img.shields.io/badge/%F0%9F%A4%97-dummy9996-FFD21E?style=flat-square
[gh-rockerBOO]: https://img.shields.io/badge/%F0%9F%A4%97-rockerBOO-FFD21E?style=flat-square
[gh-Kijai]: https://img.shields.io/badge/%F0%9F%A4%97-Kijai-FFD21E?style=flat-square
[gh-AX1Y2JP]: https://img.shields.io/badge/%F0%9F%A4%97-AX1Y2JP-FFD21E?style=flat-square
[gh-tsolful]: https://img.shields.io/badge/%F0%9F%A4%97-tsolful-FFD21E?style=flat-square
[gh-realrebelai]: https://img.shields.io/badge/%F0%9F%A4%97-realrebelai-FFD21E?style=flat-square
[gh-rzgar]: https://img.shields.io/badge/%F0%9F%A4%97-rzgar-FFD21E?style=flat-square
[gh-larryvrh]: https://img.shields.io/badge/%F0%9F%A4%97-larryvrh-FFD21E?style=flat-square
[gh-drbaph]: https://img.shields.io/badge/%F0%9F%A4%97-drbaph-FFD21E?style=flat-square
[gh-vantagewithai]: https://img.shields.io/badge/%F0%9F%A4%97-vantagewithai-FFD21E?style=flat-square
[gh-Mamad8]: https://img.shields.io/badge/%F0%9F%A4%97-Mamad8-FFD21E?style=flat-square
[gh-NicoLab28]: https://img.shields.io/badge/%F0%9F%A4%97-NicoLab28-FFD21E?style=flat-square
[gh-lightx2v]: https://img.shields.io/badge/%F0%9F%A4%97-lightx2v-FFD21E?style=flat-square
[gh-lihaoyun6]: https://img.shields.io/badge/%F0%9F%A4%97-lihaoyun6-FFD21E?style=flat-square
[gh-tutututututu]: https://img.shields.io/badge/%F0%9F%A4%97-tutututututu-FFD21E?style=flat-square
[gh-t8star]: https://img.shields.io/badge/%F0%9F%A4%97-t8star-FFD21E?style=flat-square
[gh-abakanai]: https://img.shields.io/badge/%F0%9F%A4%97-abakanai-FFD21E?style=flat-square
[gh-Winnougan]: https://img.shields.io/badge/%F0%9F%A4%97-Winnougan-FFD21E?style=flat-square
[gh-unsloth]: https://img.shields.io/badge/%F0%9F%A4%97-unsloth-FFD21E?style=flat-square
[gh-MarxistLeninist]: https://img.shields.io/badge/%F0%9F%A4%97-MarxistLeninist-FFD21E?style=flat-square
[gh-joyfox]: https://img.shields.io/badge/%F0%9F%A4%97-joyfox-FFD21E?style=flat-square
[gh-smhfacct]: https://img.shields.io/badge/%F0%9F%A4%97-smhfacct-FFD21E?style=flat-square
[gh-infosave]: https://img.shields.io/badge/%F0%9F%A4%97-infosave-FFD21E?style=flat-square
[gh-InstantX]: https://img.shields.io/badge/%F0%9F%A4%97-InstantX-FFD21E?style=flat-square
[gh-Merserk]: https://img.shields.io/badge/%F0%9F%A4%97-Merserk-FFD21E?style=flat-square
[gh-molbal]: https://img.shields.io/badge/%F0%9F%A4%97-molbal-FFD21E?style=flat-square
[gh-mvp--lab]: https://img.shields.io/badge/%F0%9F%A4%97-mvp--lab-FFD21E?style=flat-square
[gh-TenStrip]: https://img.shields.io/badge/%F0%9F%A4%97-TenStrip-FFD21E?style=flat-square
[gh-leejet]: https://img.shields.io/badge/%F0%9F%A4%97-leejet-FFD21E?style=flat-square
[gh-fal]: https://img.shields.io/badge/%F0%9F%A4%97-fal-FFD21E?style=flat-square
[gh-matlod]: https://img.shields.io/badge/%F0%9F%A4%97-matlod-FFD21E?style=flat-square
[gh-Inner--Reflections]: https://img.shields.io/badge/%F0%9F%A4%97-Inner--Reflections-FFD21E?style=flat-square
[gh-bghira]: https://img.shields.io/badge/%F0%9F%A4%97-bghira-FFD21E?style=flat-square

[badge-bf16]: https://img.shields.io/badge/bf16-0077cc?style=flat-square
[badge-fp16]: https://img.shields.io/badge/fp16-0077cc?style=flat-square
[badge-fp32]: https://img.shields.io/badge/fp32-6c757d?style=flat-square
[badge-fp8]: https://img.shields.io/badge/fp8-28a745?style=flat-square
[badge-mxfp8]: https://img.shields.io/badge/mxfp8-20c997?style=flat-square
[badge-int8]: https://img.shields.io/badge/int8-17a2b8?style=flat-square
[badge-int4]: https://img.shields.io/badge/int4-ffc107?style=flat-square
[badge-nvfp4]: https://img.shields.io/badge/nvfp4-6f42c1?style=flat-square
[badge-w4a8]: https://img.shields.io/badge/W4A8-ffc107?style=flat-square
[badge-w4a4]: https://img.shields.io/badge/W4A4-e05d44?style=flat-square
[badge-nf4]: https://img.shields.io/badge/NF4-ffc107?style=flat-square
[badge-Q2_K]: https://img.shields.io/badge/Q2__K-e05d44?style=flat-square
[badge-Q3_K_M]: https://img.shields.io/badge/Q3__K__M-fe7d37?style=flat-square
[badge-Q3_K_S]: https://img.shields.io/badge/Q3__K__S-fe7d37?style=flat-square
[badge-Q4_0]: https://img.shields.io/badge/Q4__0-dfb317?style=flat-square
[badge-Q4_1]: https://img.shields.io/badge/Q4__1-dfb317?style=flat-square
[badge-Q4_K_M]: https://img.shields.io/badge/Q4__K__M-dfb317?style=flat-square
[badge-Q4_K_S]: https://img.shields.io/badge/Q4__K__S-dfb317?style=flat-square
[badge-Q5_0]: https://img.shields.io/badge/Q5__0-97c00f?style=flat-square
[badge-Q5_1]: https://img.shields.io/badge/Q5__1-97c00f?style=flat-square
[badge-Q5_K_M]: https://img.shields.io/badge/Q5__K__M-97c00f?style=flat-square
[badge-Q5_K_S]: https://img.shields.io/badge/Q5__K__S-97c00f?style=flat-square
[badge-Q6_K]: https://img.shields.io/badge/Q6__K-0077cc?style=flat-square
[badge-Q8_0]: https://img.shields.io/badge/Q8__0-28a745?style=flat-square
[badge-UD-Q2_K_XL]: https://img.shields.io/badge/UD-Q2__K__XL-e05d44?style=flat-square
[badge-UD-Q3_K_XL]: https://img.shields.io/badge/UD-Q3__K__XL-fe7d37?style=flat-square
[badge-IQ1_S]: https://img.shields.io/badge/IQ1__S-b02a37?style=flat-square
[badge-IQ1_M]: https://img.shields.io/badge/IQ1__M-d64545?style=flat-square
[badge-noinfo]: https://img.shields.io/badge/no%20description-6c757d?style=flat-square&logoColor=white

[cat-cond]: https://img.shields.io/badge/Conditioning-0077cc?style=flat-square
[cat-prompt]: https://img.shields.io/badge/Prompt-28a745?style=flat-square
[cat-upscale]: https://img.shields.io/badge/Upscaling-fe7d37?style=flat-square
[cat-accel]: https://img.shields.io/badge/Acceleration-6f42c1?style=flat-square
[cat-port]: https://img.shields.io/badge/Port-17a2b8?style=flat-square
[cat-face]: https://img.shields.io/badge/Face%20Refine-e83e8c?style=flat-square
