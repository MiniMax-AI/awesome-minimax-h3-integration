# MiniMax H3 Integration Guide

A short guide to the official MiniMax H3 model, deployment paths, and documentation.

## Official links

| Resource | Link |
| :--- | :--- |
| Model repository | [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) |
| Model card and checkpoints | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| Video generation API | [MiniMax Platform](https://platform.minimax.io/docs/api-reference/video-generation-v2-create) |
| Prompt guide: Base | [Read the guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md) |
| Prompt guide: Reference | [Read the guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md) |

## What H3 supports

MiniMax H3 accepts text, images, video, and audio, and generates video with native stereo audio. The official release supports clips from 4 to 15 seconds, 24 FPS output, and resolutions up to 2K through the full workflow.

| Model | Use it for |
| :--- | :--- |
| H3-Base-FL2VA | Text-to-video, first-frame-to-video, last-frame-to-video, and first-and-last-frame-to-video. |
| H3-Base-Ref2VA | Video generation from text plus image, video, and audio references. |
| H3-Context-IR | Preparing complex multimodal context before generation. |
| H3-Regenerate-2K | Regenerating an H3 result at 2K. |

For the complete input limits and workflow details, see the [official system overview](https://github.com/MiniMax-AI/MiniMax-H3#system-overview).

## Run H3

The official repository documents four supported paths:

| Runtime | Official guide |
| :--- | :--- |
| SGLang | [MiniMax H3 cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3) |
| vLLM | [MiniMax H3 recipes](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3) |
| Diffusers | [MiniMax H3 documentation](https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md) |
| ComfyUI | [MiniMax H3 tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) |

To download the original checkpoints for SGLang or vLLM:

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --include "model_index.json" "FL2VA/*" "Ref2VA/*" \
  --local-dir MiniMax-H3
```

For Diffusers, follow the [official loading instructions](https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md). For a reproducible local deployment, start with the [official H3-Base workflow](https://github.com/MiniMax-AI/MiniMax-H3#local-deployment-of-h3-base).

## Prompting

Use the official prompt guides before building a workflow:

- [Base prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- [Reference prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [Official H3 prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing)

## License

MiniMax H3 is released under the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).

## Contact

For questions about MiniMax H3, contact [model@minimax.io](mailto:model@minimax.io).

## Thanks

This guide follows the [official MiniMax H3 repository](https://github.com/MiniMax-AI/MiniMax-H3) and links to the official documentation maintained by MiniMax and its runtime partners.
