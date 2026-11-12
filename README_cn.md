<p align="center">
  <a href="https://github.com/MiniMax-AI/MiniMax-H3"><img src="https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/assets/minimax-h3-header.gif" alt="MiniMax H3"></a>
</p>

# MiniMax H3 Integrations

[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) 和 [MiniMax-H3-Image](https://github.com/MiniMax-AI/MiniMax-H3-Image) 的模型、工具和项目索引，由社区共同维护。

[English](README.md) | 简体中文

## 目录

- [MiniMax H3](#minimax-h3)
  - [官方资源](#h3-official) · [量化](#h3-quantizations) · [组件](#h3-components) · [加速与蒸馏](#h3-acceleration) · [LoRA 与微调](#h3-loras) · [推理引擎](#h3-runtimes) · [应用与工作室](#h3-apps) · [API 集成](#h3-api) · [ComfyUI 节点](#h3-comfyui) · [工作流](#h3-workflows) · [提示词与 Agent 技能](#h3-prompting) · [训练](#h3-training) · [部署与托管](#h3-serving) · [指南](#h3-guides)
- [MiniMax-H3-Image](#minimax-h3-image)
  - [官方资源](#img-official) · [量化与格式转换](#img-quantizations) · [蒸馏与适配器](#img-adapters) · [推理引擎](#img-runtimes) · [应用](#img-apps) · [API 集成](#img-api) · [ComfyUI 节点与工作流](#img-comfyui) · [训练](#img-training) · [提示词](#img-prompting) · [评测](#img-benchmarks)
- [贡献](#贡献)

## MiniMax H3

MiniMax H3 是 MiniMax 开放权重的音视频生成模型：以文字、图片、视频和音频为上下文，生成 4 到 15 秒、最高 2K、带原生立体声的视频。它有两个版本：FL2VA 根据文字和可选的首尾帧生成，Ref2VA 最多可参考 9 张图片、3 段视频和 3 段音频。

<a id="h3-official"></a>

### 官方资源

<table>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/MiniMax-H3">MiniMax-H3</a></td><td>官方仓库：参考实现和提示词指南。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">MiniMaxAI/MiniMax-H3</a></td><td>原始权重，包括 FL2VA（首尾帧）和 Ref2VA（多参考）两个版本，以及文本编码器、视频 VAE 和音频 VAE。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/630433fea362e7e8b51f5e2b/bsDc8lR72KhKvm-wXxYsv.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Comfy-Org/MiniMax-H3">Comfy-Org/MiniMax-H3</a></td><td>ComfyUI 单文件权重，包括剪枝后的 BF16、FP8 和 INT8 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://docs.comfy.org/tutorials/video/minimax/minimax-h3">ComfyUI tutorial</a></td><td>ComfyUI 官方 H3 教程，见 <a href="https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui">首发公告</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json">ComfyUI workflow templates</a></td><td>ComfyUI 自带的官方模板：文生视频、<a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json">图生视频</a>和<a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json">参考生视频</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md">Video prompt writing guides</a></td><td>官方提示词指南：基础模式（FL2VA）和<a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md">参考模式（Ref2VA）</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://platform.minimax.io/docs/api-reference/video-generation-v2-create">MiniMax API</a></td><td>官方视频生成 API；中国大陆请用 <a href="https://platform.minimaxi.com/docs/api-reference/video-generation-v2-create">platform.minimaxi.com</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://hailuoai.video/tools/minimax-h3">Hailuo AI</a></td><td>在网页上或<a href="https://hub.minimax.io/">桌面应用</a>里直接使用 H3；中国大陆请用 <a href="https://hailuoai.com/">hailuoai.com</a> 和 <a href="https://hub.minimaxi.com/">hub.minimaxi.com</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://docs.comfy.org/tutorials/partner-nodes/minimax/minimax-h3">ComfyUI API nodes</a></td><td>ComfyUI 自带的 API 节点，支持文生视频、首尾帧和参考生视频，通过 Comfy 账号调用 H3 和 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/cli">MiniMax CLI</a></td><td>MiniMax API 的官方命令行工具，可直接生成 H3 视频。</td></tr>
</table>

<a id="h3-quantizations"></a>

### 量化

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62ecdc18b72a69615d6bd857/E4lkPz1TZNLzIFr_dR273.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/unsloth/MiniMax-H3-GGUF">unsloth/MiniMax-H3-GGUF</a></td><td>剪枝版 FL2VA 和 Ref2VA 的 GGUF，从 Q2_K（6.26 GiB）到 Q8_0，另有 Qwen3-VL 文本编码器的 GGUF。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/Yy9XdzixLpDyVyN4fUvP5.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants">DmitryDB/MiniMax-H3-ComfyUI-Quants</a></td><td>FL2VA 和 Ref2VA 的 INT8 ConvRot 和 NVFP4 版本，10.86 到 21.91 GiB，附 VAE 文件。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/Yy9XdzixLpDyVyN4fUvP5.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV">DmitryDB/MiniMax-H3-DynTime-sQKV</a></td><td>INT8 DT-sQKV 版本，需要配合该仓库的 ComfyUI 运行时补丁。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63150d9f8be84e1ab412890e/slVOxsRI070drG6kICymD.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/rockerBOO/minimax-h3-nvfp4-convrot">rockerBOO/minimax-h3-nvfp4-convrot</a></td><td>面向 Blackwell 显卡的 NVFP4 FL2VA 和 Ref2VA，推荐 INT8 ConvRot 版；另有给其他显卡用的实验性 INT4 ConvRot FL2VA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot">Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot</a></td><td>剪枝版 NVFP4（11.67 GiB）、INT8 ConvRot 和 INT4/INT8 混合 ConvRot 版本。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6357da9b68659b265eb84aab/8PEgQsPKTJVQQD2_87mP9.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/tsolful/Minimax_H3_INT4MixedConvRot">tsolful/Minimax_H3_INT4MixedConvRot</a></td><td>FL2VA 和 Ref2VA 的剪枝版 INT4/INT8 混合 ConvRot。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/b3c31e18f3b7cc1c5c059058221195dc.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot">Merserk/MiniMax-H3-INT4-ConvRot</a></td><td>INT4 ConvRot，完整版 16.67 GiB，剪枝版 10.56 GiB。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/357a2cb738d2a9ad5c01dd6296792c22.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/AX1Y2JP/MiniMax-H3-W4A8-ConvRot">AX1Y2JP/MiniMax-H3-W4A8-ConvRot</a></td><td>剪枝版 W4A8 ConvRot DiT（11.68 GiB）和 W4A8 文本编码器。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63297908f0b2fc94904a65b8/hr-fUccqYPlTfmfKgoZAO.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Kijai/MiniMax-H3-experimental">Kijai/MiniMax-H3-experimental</a></td><td>实验性的 W4A8 混合精度 DiT 和 INT8 ConvRot 视频 VAE。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62c3f1dd73c3ba0c801033ce/3STHHoSpTLGI-Alcpf7Cg.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI">Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI</a></td><td>剪枝版 W4A8 ConvRot DiT，附配套的量化文本编码器。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/644717ea27a3e9e0b7c7207a/aNn8akdZfegf46fkZ4R9h.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn">rzgar/minimax_h3_fl2va_fp8_e4m3fn</a></td><td>FL2VA 的 FP8 E4M3FN、MXFP8，以及 FP8 加 FP16 注意力版本。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/6701ed70ea8cd899201c524ab3ccac8c.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/abakanai/Minimax_h3_hybrid">abakanai/Minimax_h3_hybrid</a></td><td>面向 Blackwell 显卡的混合 NVFP4 Ref2VA 版本，14.03 和 16.38 GiB。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672c8f0070734f59f146b73c/5lGMq-_Patlincxq97jf5.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4">DiffSynth-Studio/MiniMax-H3-NF4</a></td><td>给 DiffSynth-Studio 用的 NF4 DiT、文本编码器和 VAE，8 GB 显存起步。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1661165878439-noauth.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4">WaveCut/MiniMax-H3-OrbitQuant-W4A4</a></td><td>W4A4 版本，附量化文本编码器；需要 ComfyUI-OrbitQuant 节点。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-GGUF">Abiray/MiniMax-H3-GGUF</a></td><td>完整版 FL2VA 和 Ref2VA 的 GGUF，Q3_K_S 到 Q8_0，附文本编码器量化和工作流。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF">Abiray/MiniMax-H3-Pruned-GGUF</a></td><td>剪枝版 FL2VA 的 GGUF，Q3_K_M 和 Q8_0。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/68761990332d15464ccc8dee/MaE3RLtzT8Q3tQ5I-n38s.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/realrebelai/MiniMax-H3_GGUFs">realrebelai/MiniMax-H3_GGUFs</a></td><td>FL2VA 和 Ref2VA 的 Q3_K_M、Q4_K_M GGUF，另有 Q2_K 和 Q4_K_M 文本编码器，Q2_K 只有 7.91 GiB。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/67b808c770117114270d902e/dLdwkNZowSMsqTZ3Znfg2.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/vantagewithai/MiniMax-H3-comfyUI-GGUF">vantagewithai/MiniMax-H3-comfyUI-GGUF</a></td><td>给 ComfyUI 用的完整版 FL2VA 和 Ref2VA GGUF，Q3_K_M 到 Q8_0。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/2bfb8e5718e39dd319e9320abab557c1.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF">MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF</a></td><td>IQ1_S（3.78 GiB）和 IQ1_M 版本，是目前最小的 DiT。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/638a3f5a8d7cce863e70a91b/Z_KI0kdABbpRkXV51JpOG.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/molbal/MiniMax-H3-GGUF">molbal/MiniMax-H3-GGUF</a></td><td>剪枝版 Ref2VA 的 GGUF，包括手工调过的混合精度布局。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/92b2fbafe275bf1c126d5a687d6a7f2f.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/leejet/MiniMax-H3-GGUF">leejet/MiniMax-H3-GGUF</a></td><td>剪枝版 FL2VA 的 GGUF，Q2_K 和 Q4_K_M。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65167486c1ec63624f5e45dd/zJlq2UH7TzeLv77AsbFUn.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/joeygambino/MiniMax-H3-curve-GGUF">joeygambino/MiniMax-H3-curve-GGUF</a></td><td>剪枝版 FL2VA 和 Ref2VA 的 GGUF，Q4_0、Q5_1 和 Q8_0；需要给 ComfyUI-GGUF 打补丁的 H3-Multishot 节点包。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/68a41bb168b64003eab9cfa4/Gq5_H6FGLr1k3hSsshdfa.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/jfar-z/MiniMax-H3-FL2VA-Ref2VA-Hybrid-NVFP4">jfar-z/MiniMax-H3-FL2VA-Ref2VA-Hybrid-NVFP4</a></td><td>smhfacct 的 FL2VA 与 Ref2VA 混合方案的 NVFP4 版本，四种层段组合，每个 11.67 GiB，面向 Blackwell 显卡。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/cac9874103e7c159dc5eb6ec977f3ceb.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4">lilcheaty/MiniMax-H3-NVFP4</a></td><td>剪枝版 FL2VA 和 Ref2VA 的 NVFP4 版本，各 11.67 GiB，另有完整和混合精度的 Ref2VA 版本，面向 Blackwell 显卡。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/42b9a338c1a323b57ce3b506de76f8ae.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/DeepBeepMeep/MiniMax-H3">DeepBeepMeep/MiniMax-H3</a></td><td>WanGP 使用的 H3 系列权重：FL2VA、Ref2VA、FastH3 Omni、HyperFlow Flash、Fun ControlNet 和 H3-Image，有 BF16 和 INT8 版本。</td></tr>
</table>

<a id="h3-components"></a>

### 组件

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1650342890988-623c6253389748c9f72ca287.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union">alibaba-pai/MiniMax-H3-Fun-Controlnet-Union</a></td><td>一个控制分支同时支持 Canny、深度、HED、MLSD、姿态控制和视频局部重绘。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/NCer_WlbHMkH1ZU8Q_I8S.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lihaoyun6/MiniMax-H3-Ref-Patch">lihaoyun6/MiniMax-H3-Ref-Patch</a></td><td>148 MiB 的补丁，让 FL2VA 更接近 Ref2VA 的效果；需要配套的 ComfyUI 节点。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63297908f0b2fc94904a65b8/hr-fUccqYPlTfmfKgoZAO.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Kijai/MiniMax-H3-TAE">Kijai/MiniMax-H3-TAE</a></td><td>9 MiB 的小型自编码器，用于 ComfyUI-KJNodes 里的预览。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/2a4082bfe6d2d4d206dbd655197388f0.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE">Mamad8/MiniMax-H3-Image-VAE</a></td><td>实验性的单帧 VAE，用视频模型出静态图；生成视频时仍用原版 VAE。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/67c576e958e13035e6e4b0d5/mECPgQ0bhniFavYJPDpV4.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler">LBH-123-AI/Minimax_h3_latent_Upscaler</a></td><td>学习型潜空间放大模型，1.0 到 4.0 倍，配合它的 ComfyUI 节点使用。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62b4b5beb25cb80fcf278354/SIddx3hYu5rWXA4-O3Oaj.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K">Open-H3-Regenerate-2K</a></td><td>NVIDIA 的三步精修模型，把低分辨率的 H3 草稿提升到 2560x1440，附<a href="https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K">代码</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6767c79846bb6707b92e5fb2/lqjSqaUQW3WEMVUG6KFu_.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/ZawShiShawn/Gestura-MiniMax-H3-VAE-LiteRT">ZawShiShawn/Gestura-MiniMax-H3-VAE-LiteRT</a></td><td>LiteRT 格式的 H3 VAE，给在安卓手机上本地生成视频的 Gestura 应用使用。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/7ddf7fcf26e8da8d32da7a3c4e73af52.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE">speach1sdef178/MiniMax-H3-X2-Detail-VAE</a></td><td>解码出两倍分辨率的视频 VAE；需要 ComfyUI-MiniMaxH3_LatentUpscaler 里的快速 VAE 节点。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62b4b5beb25cb80fcf278354/SIddx3hYu5rWXA4-O3Oaj.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC">Efficient-Large-Model/Open-H3-Regenerate-2K-IC</a></td><td>NVIDIA 精修模型的上下文 LoRA 版本，借 Ref2VA 把 768p 草稿重新生成为 1080p、2K 或 4K。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65167486c1ec63624f5e45dd/zJlq2UH7TzeLv77AsbFUn.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/joeygambino/MiniMax-H3-encoder-GGUF">joeygambino/MiniMax-H3-encoder-GGUF</a></td><td>Qwen3-VL-32B 文本编码器的 GGUF 版本，给 ComfyUI 用。</td></tr>
</table>

<a id="h3-acceleration"></a>

### 加速与蒸馏

<table>
<tr><td width="56"><img src="https://github.com/ModelTC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ModelTC/Minimax-H3-Turbo">Minimax-H3-Turbo</a></td><td>LightX2V 的 4 步和 8 步蒸馏 LoRA，<a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo">权重在 Hugging Face</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63783918d925e7dad20748a3/SIMpPx2nIKyEF7Mx9E3hC.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora">larryvrh/MiniMax-H3-Turbo-Lora</a></td><td>社区重训的 4 步 Turbo LoRA，常用默认版本是 <code>v4_step600_ema</code>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63473b59e5c0717e6737b872/e0rwJa886MD1hmdeFr9Wh.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI">drbaph/MiniMax-H3-Turbo-Lora-ComfyUI</a></td><td>适配 ComfyUI 剪枝版权重的 Turbo LoRA，包括压到 rank 21 的版本。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI">Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI</a></td><td>适配剪枝版 ComfyUI 权重的 Turbo LoRA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63297908f0b2fc94904a65b8/hr-fUccqYPlTfmfKgoZAO.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Kijai/MiniMax-H3_comfy">Kijai/MiniMax-H3_comfy</a></td><td>ComfyUI 格式的 LightX2V Turbo LoRA，完整版和 rank 21 版。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/637745113a63a2983ffbde13/wZKoPSWQKpe2JpFsX3nLa.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/InstantX/MiniMax-H3-Turbo-Lora-Diffusers">InstantX/MiniMax-H3-Turbo-Lora-Diffusers</a></td><td>diffusers 格式的 4 步 Turbo LoRA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6864d1f250596cb028f66be3/TOr9t2xU3PJFjSXGQvTYK.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/joyfox/MiniMax-H3-Turbo">joyfox/MiniMax-H3-Turbo</a></td><td>4 步 FL2VA LoRA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/674ef761b783a2a16788d7fe/cxPCVoNPpnehsDwy3bg14.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA">Tutu 20-to-8 NFE LoRA</a></td><td>把 20 次推理压到 8 次的音视频 LoRA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/657872063a56e4034e3cb5e2/MLAv_Dm8jiXY5f0apfUVP.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/t8star/minimax-h3-4step-turbo-loras-comfyui-exp">t8star/minimax-h3-4step-turbo-loras-comfyui-exp</a></td><td>实验性的 INT8 ConvRot Turbo LoRA，给 ComfyUI 用。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/693c433f38397e97c1c9fb78/VNwN7gvr2RGpxeBUIHNqT.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf">infosave/MiniMax-H3-Turbo-cmf</a></td><td>CMF 格式的 Turbo 权重，Q4TP 和 Q2TP。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1650342890988-623c6253389748c9f72ca287.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs">alibaba-pai/MiniMax-H3-Acc-LoRAs</a></td><td>PDD 8 步加速 LoRA，FL2VA 和 Ref2VA 各一个，新版 ComfyUI 可以直接加载。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/1af3b10983f16ee56a7ad7b48c1e5df7.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/aptech0081/MiniMax-H3-PDD-ComfyUI">aptech0081/MiniMax-H3-PDD-ComfyUI</a></td><td>转成 ComfyUI 格式的 PDD LoRA，另有合入 PDD 的 INT8 Ref2VA 权重。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/657872063a56e4034e3cb5e2/MLAv_Dm8jiXY5f0apfUVP.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/t8star/Minimax-H3-Super-Acceleration-Comfy">t8star/Minimax-H3-Super-Acceleration-Comfy</a></td><td>为 T8 音频节点重新打包的 PDD LoRA。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/ce52788c884be3d7654e823007579f5a.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/1ronman1993/MiniMax-H3-SVDQuant-fp4-pdd8">1ronman1993/MiniMax-H3-SVDQuant-fp4-pdd8</a></td><td>已经合入 PDD 8 步 LoRA 的 SVDQuant FP4 版本，另有 <a href="https://huggingface.co/1ronman1993/MiniMax-H3-SVDQuant-int4-pdd8">INT4 版本</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree">FastH3 4-step Preview</a></td><td>FastVideo 的 4 步免数据蒸馏权重，带稀疏注意力；也有 <a href="https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA">LoRA 版</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2">FastH3 8-Step V2</a></td><td>8 步文生音视频权重，稀疏注意力的稀疏度为 80%。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4">FastH3 Trim NVFP4</a></td><td>实验性的 8 步模型，保留 50 层中的 42 层，NVFP4 共 11.1 GiB，面向 Blackwell 和 DGX Spark。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1">FastH3-Omni 8-Step</a></td><td>8 步的多参考音视频生成，可用图片、视频和音频作参考。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63565cc56d7fcf1bedb7d347/g6wWXLryVUwXcaKTuRerh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy">FastVideo/FastVideo-FastH3-Comfy</a></td><td>FastH3 8 步 V2 的 ComfyUI 打包版。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/68761990332d15464ccc8dee/MaE3RLtzT8Q3tQ5I-n38s.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/realrebelai/FastH3_GGUFs">realrebelai/FastH3_GGUFs</a></td><td>FastH3 4 步的 GGUF 转换版。</td></tr>
<tr><td width="56"><img src="https://github.com/Pika-Labs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Pika-Labs/Pika-H3-Flex">Pika H3 Flex</a></td><td>Pika 的 6 步 FL2VA 和 Ref2VA 适配器，原生输出 480p 和 768p，附命令行、diffusers 和 ComfyUI 示例。</td></tr>
<tr><td width="56"><img src="https://github.com/vita-epfl.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vita-epfl/PTD-inference/tree/release/comfyui-inference">PTD</a></td><td>EPFL 的 4 步和 6 步轨迹蒸馏，附<a href="https://huggingface.co/epfl-vita/PTD-weights">权重</a>和 ComfyUI 集成。</td></tr>
<tr><td width="56"><img src="https://github.com/Yzmblog.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Yzmblog/DMAD">DMAD</a></td><td>4 步和 8 步文生音视频 LoRA，附<a href="https://huggingface.co/ZhengmingYu/DMAD">权重</a>、工作流和训练代码。</td></tr>
<tr><td width="56"><img src="https://github.com/OpenVDN.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/OpenVDN/vdn-minimax-h3">VDN</a></td><td>Video Delta Net：线性和 softmax 混合注意力，每步快 2.6 到 2.9 倍，附<a href="https://huggingface.co/OpenVDN/vdn-minimax-h3">权重</a>和训练代码。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63473b59e5c0717e6737b872/e0rwJa886MD1hmdeFr9Wh.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/drbaph/vdn-minimax-h3-int8-convrot-comfyui">drbaph/vdn-minimax-h3-int8-convrot-comfyui</a></td><td>ComfyUI 格式的 INT8 ConvRot VDN。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/657872063a56e4034e3cb5e2/MLAv_Dm8jiXY5f0apfUVP.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy">t8star/Vdn-Minimax-H3-Comfy</a></td><td>配合 T8 节点的 VDN 模型包，放好即可用。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/66213d7f118a0a5cef880b5e/hr3ia85xax2nyW0OO0F7D.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA">RAVEN Streaming LoRA</a></td><td>4 步预览版，把 H3 变成逐段生成的流式模型，配合 <a href="https://github.com/mvp-ai-lab/RAVEN">RAVEN</a> 代码使用。</td></tr>
<tr><td width="56"><img src="https://www.gravatar.com/avatar/7b0abe8d2e4aff190133f25f13729f6b?d=retro&size=100" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot">MATLOWAI/minimax-h3-fused-turbo-int8-convrot</a></td><td>单个 21 GB 的 INT8 ConvRot 文件，已合入 Turbo 和运动 LoRA，在 ComfyUI 里 4 步完成文生视频、图生视频和 Ref2VA。</td></tr>
<tr><td width="56"><img src="https://github.com/Video-Rebirth.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Video-Rebirth/hyperflow-h3-flash">HyperFlow H3 Flash</a></td><td>Video Rebirth 压缩到 13.7B 的 H3，4 到 8 步生成带立体声的视频，有 INT8 和 GGUF <a href="https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash">权重</a>；另有给完整版 H3 用的 <a href="https://huggingface.co/VideoRebirthAI/hyperflow">8 步 LoRA</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/TaoLiveAIGC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/TaoLiveAIGC/TaoMate-H3">TaoMate-H3</a></td><td>让 H3 三步流式生成，附<a href="https://huggingface.co/TaoLiveAIGC/TaoMate-H3">权重</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/thu-ml.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/thu-ml/TurboDiffusion">TurboDiffusion</a></td><td>面向 H3 FL2VA 的 W8A8 算子、采样器和多卡运行时，附 TurboH3-FL2VA-I2VA 权重。</td></tr>
<tr><td width="56"><img src="https://github.com/NVlabs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NVlabs/LongLive">LongLive</a></td><td>NVIDIA 的 LongLive-Plug 蒸馏，附 H3 的<a href="https://huggingface.co/Efficient-Large-Model/LongLive-Plug-MiniMax-H3-few-step">少步数</a>和 <a href="https://huggingface.co/Efficient-Large-Model/LongLive-Plug-MiniMax-H3-cfg">CFG</a> 适配器。</td></tr>
<tr><td width="56"><img src="https://github.com/NVlabs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NVlabs/FastGen">FastGen</a></td><td>NVIDIA 的蒸馏框架，支持的音视频模型里有 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/vipshop.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vipshop/cache-dit">cache-dit</a></td><td>diffusers 的缓存加速和并行推理，支持 H3 的 T2VA、FL2VA 和 Ref2VA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6568921511b2bbd6c2e9841c/v-pfaTZmvAiLeBjS2-eDb.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview">Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview</a></td><td>学习型稀疏注意力，各模式下只保留约 10% 的注意力块，另有专门微调的 <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview">R2VA 预测器</a>；可在 ComfyUI 或 Miowtion 里运行。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/680de13385293771bc57400b/qnjz3avU_cKzRZA2aGjuZ.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA">lightx2v/Minimax-h3-Turbo-SLA</a></td><td>4 步 FL2VA LoRA，加入稀疏度 85% 的稀疏线性注意力，在 LightX2V 里用 RTX 5090 约快 2.5 倍，另有 ComfyUI 版本。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6ad5e8ef64cf8fd69bf5f843/28JY9Lma4ZOqJ8GgMZduD.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-LoRAs">taobao-content-ai/MiniMax-H3-PTD-LoRAs</a></td><td>每一步各用一个 PTD LoRA，让 Ref2VA 4 步或 8 步出片，附 diffusers 代码和 ComfyUI 节点，节点需要 ComfyUI-PTD 和未剪枝的 Ref2VA 模型。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/f72b4505394ad4d2258dbba49ade6e85.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/pdmd2026/pdmd_4NFE_lora">pdmd2026/pdmd_4NFE_lora</a></td><td>用投影分布匹配蒸馏出的 4 步 LoRA，rank 128，diffusers 键名格式。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6a550d1dc84594e3ada10828/BLGmI3tqOxQmO1ti1rxUH.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF">ChrisColeTech/minimax-h3-turbo-GGUF</a></td><td>合入 LightX2V Turbo LoRA 的剪枝版 FL2VA 和 Ref2VA GGUF，另附配合 ComfyUI-ClipProj 的 4B 和 8B 文本编码器。</td></tr>
</table>

<a id="h3-loras"></a>

### LoRA 与微调

<table>
<tr><td width="56"><img src="https://huggingface.co/avatars/a0fe0c3e0ee676535c5cae05e05aacb7.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/matlod/minimax-h3-turnaround">minimax-h3-turnaround</a></td><td>从一张参考图生成同一主体的五个旋转视角。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6380ebb8471a4550ff255c62/JLWd_lQkzQN81FVIva1w1.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA">Realism - People</a></td><td>fal 训练的 LoRA，让日常场景里的人物更自然。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/a7bf1af19e638162579666509c490220.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime">Looping Sketch Anime</a></td><td>手绘 2D 动画风格，可以无缝循环。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/642b468deb31218a5f1e69f7/FWS9Yc-nckXsKBr5l-pij.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/TenStrip/Krea2-H3-Style-Lora">Krea2 Style</a></td><td>实验性风格 LoRA，向 Krea 2 的画风靠拢，只对文生视频有效。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/39c999386889182c4ac74c6c612f8bcf.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models">smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models</a></td><td>以 FL2VA 为基础、合入 Ref2VA 的 AdaLN 层，有四种层段组合。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/657872063a56e4034e3cb5e2/MLAv_Dm8jiXY5f0apfUVP.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/t8star/minimax_h3_ref2va_patchin_hf102">t8star/minimax_h3_ref2va_patchin_hf102</a></td><td>实验性的 Ref2VA 权重修改，把一处高频分量调高 2%。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6333948fea2dcff925248fca/Bef-LRCRhI3zZzt-lY8mv.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Viggle/Viggle-Remake">Viggle Remake</a></td><td>Viggle 的两个 LoRA，替换视频里的人物，或修改物体、背景和天气；<a href="https://huggingface.co/Viggle/Viggle-Animate">Viggle Animate</a> 靠一帧重绘画面替换角色。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/653772fed7616b72795db894/1GPjjeZP_RGRl1pYgFdZc.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/PrunaAI/P-H3-Image">P-H3-Image</a></td><td>PrunaAI 基于 H3 视频模型微调的文生图和图像编辑模型。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/noauth/zYUGNDVFyLncD1OHov1pB.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI">Alissonerdx/Minimax-H3-ComfyUI</a></td><td>实验性 LoRA，用于动作迁移、角色和头部替换、风格迁移、特效编辑和动态壁纸。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/232ea307b6e3f75cffe2568a99541dad.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA">360 Orbit</a></td><td>FL2VA LoRA，把一张照片变成时间静止的 360 度环绕镜头。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/noauth/n2bexa-LzvGqmxyOGaonu.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity">Minimax-h3 Singularity</a></td><td>融合 FL2VA 和 Ref2VA 权重后再微调的模型，在 ComfyUI 里支持文生、图生、参考生和视频生视频；另有<a href="https://huggingface.co/WAS/Minimax-h3_Singularity_v2_pruned">剪枝版</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65bb837dbfb878f46c77de4c/23gZ_lBEwyoqjexFy9QLD.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp">prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp</a></td><td>实验性的 rank 16 LoRA，用于面部特写，表情和细微动作更真实；提示词以 <code>Facial Realism</code> 开头。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6928a3ca735961e729669bb3/RrdhFVLZ35EUaBSoYb4IE.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/LiconStudio/Minimax-H3-VBVR">LiconStudio/Minimax-H3-VBVR</a></td><td>用重新整理的 VBVR 数据训练的图生视频 LoRA，减少物体重复和人物与物体互相穿插。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/635dd6cd4fabde0df74aeae6/zDmJWNx95QODS2mbAAxNZ.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/alvdansen/h3-keyframe-animation">alvdansen/h3-keyframe-animation</a></td><td>三个 LoRA，按一到两张参考画保持手绘动画的节奏；个人和小企业免费，下载需同意条款。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/noauth/nJsOmzl5Jf72CxaaHjoiF.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/XGENlabs/XGEN-JING">XGENlabs/XGEN-JING</a></td><td>基于 Ref2VA 的 4 步第一人称模型，根据键盘控制的镜头、参考图和文字生成音视频。</td></tr>
<tr><td width="56"><img src="https://github.com/Junchao-cs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Junchao-cs/SolarWM">SolarWM</a></td><td>世界模型，H3 版本从首帧和提示词流式生成可控镜头的视频，附训练代码和 4 步的 SolarRenderer。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6a6748afef317ed5adb30050/Ca36mpBOmdmrgVPxtMafQ.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora">Wushu Action LoRA</a></td><td>用 924 段武打片段训练的武打动作 LoRA，触发词 <code>wushu_action</code>。</td></tr>
</table>

<a id="h3-runtimes"></a>

### 推理引擎

<table>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Comfy-Org/ComfyUI">ComfyUI</a></td><td>首发即原生支持，主线已内置 INT8。</td></tr>
<tr><td width="56"><img src="https://github.com/modelscope.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/modelscope/DiffSynth-Studio">DiffSynth-Studio</a></td><td>提供 <code>MiniMaxH3Pipeline</code>，NF4 推理 8 GB 显存起步。</td></tr>
<tr><td width="56"><img src="https://github.com/ModelTC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ModelTC/LightX2V">LightX2V</a></td><td>支持并行、量化 DiT 和特征缓存的推理框架，Turbo LoRA 也出自这里。</td></tr>
<tr><td width="56"><img src="https://github.com/NVlabs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3">Sana Sol-Engine</a></td><td>NVIDIA 针对各型号 GPU 优化的推理方案，比基线快 3.55 到 4.52 倍；另有最多 8 卡的 4 步 Sol-H3。</td></tr>
<tr><td width="56"><img src="https://github.com/hao-ai-lab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/hao-ai-lab/FastVideo">FastVideo</a></td><td>FastH3 背后的稀疏注意力和步数蒸馏框架。</td></tr>
<tr><td width="56"><img src="https://github.com/aigc-apps.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/aigc-apps/VideoX-Fun">VideoX-Fun</a></td><td>Fun ControlNet-Union 和 PDD LoRA 的训练与推理代码。</td></tr>
<tr><td width="56"><img src="https://github.com/antirez.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/antirez/h3.c">h3.c</a></td><td>苹果芯片上的原生 Metal 引擎，支持文生视频、首尾帧和多参考生成。</td></tr>
<tr><td width="56"><img src="https://github.com/drawthingsai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/drawthingsai/draw-things-community">Draw Things</a></td><td>Draw Things 应用背后的 Swift/Metal 引擎，在苹果芯片上跑 FL2VA、Ref2VA 和 FastH3 Omni。</td></tr>
<tr><td width="56"><img src="https://github.com/ddalcu.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ddalcu/mlx-serve">mlx-serve</a></td><td>苹果芯片上的应用和服务端，支持 8 步的 FastH3 Omni 多参考视频。</td></tr>
<tr><td width="56"><img src="https://github.com/komikndr.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/komikndr/raylight">Raylight</a></td><td>多卡 ComfyUI 运行时，把序列拆到多张显卡上并行计算。</td></tr>
<tr><td width="56"><img src="https://github.com/unslothai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/unslothai/unsloth">Unsloth</a></td><td>支持运行和训练 MiniMax-H3。</td></tr>
<tr><td width="56"><img src="https://github.com/fluxmatrix.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/fluxmatrix/h3-flash">h3-flash</a></td><td>面向 8 卡 B200 的推理脚本和网页演示，用序列并行和融合算子加速。</td></tr>
<tr><td width="56"><img src="https://github.com/huggingface.png?size=96" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/blog/hf-torchtpu">TorchTPU + Diffusers</a></td><td>在 Cloud TPU v6e-4 上用原生 PyTorch、<code>torch.compile</code> 和张量并行运行 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/leejet.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/leejet/stable-diffusion.cpp">stable-diffusion.cpp</a></td><td>C/C++ 推理引擎，可运行 H3 及其 GGUF 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/mudler.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/mudler/LocalAI">LocalAI</a></td><td>本地的 OpenAI 兼容服务，4.9.0 版加入了 H3 音视频生成。</td></tr>
<tr><td width="56"><img src="https://github.com/ztxz16.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ztxz16/fastllm">fastllm</a></td><td>原生支持 H3 文生音视频，附 HTTP 演示。</td></tr>
<tr><td width="56"><img src="https://github.com/0xShug0.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/0xShug0/audio.cpp">audio.cpp</a></td><td>C++ 引擎，用 Q4_K GGUF 运行 H3 文生音视频，DiT 可选 INT8 ConvRot。</td></tr>
<tr><td width="56"><img src="https://github.com/1CatAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/1CatAI/1Cat-vLLM">1Cat-vLLM</a></td><td>面向 V100 的 vLLM 分支，原生支持 H3 音视频生成。</td></tr>
<tr><td width="56"><img src="https://github.com/gufo-org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gufo-org/gufo">gufo</a></td><td>面向 AMD Strix Halo 的推理引擎，可运行 H3 有声视频。</td></tr>
<tr><td width="56"><img src="https://github.com/localai-org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/localai-org/vllm.cpp">vllm.cpp</a></td><td>不依赖 Python 的 C++ 推理引擎，用 Q4_K_M GGUF 通过 <code>/v1/videos</code> 提供 H3 服务，在一台 Jetson Thor 上实测每步 34.6 秒。</td></tr>
</table>

<a id="h3-apps"></a>

### 应用与工作室

<table>
<tr><td width="56"><img src="https://github.com/agent-next.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/agent-next/video-agent">open-video</a></td><td>用 <code>install</code>、<code>pull</code>、<code>run</code> 三条命令在自己的显卡上跑 H3，另附让编程 agent 生成视频的技能。</td></tr>
<tr><td width="56"><img src="https://github.com/FlashML-org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/FlashML-org/FreeVideo">FreeVideo</a></td><td>基于 VDN-H3 的本地 ComfyUI 工作台，支持 8 GB 显存。</td></tr>
<tr><td width="56"><img src="https://github.com/D3voz.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/D3voz/openjutsu">Openjutsu</a></td><td>用 SAM 3.1 跟踪替换视频里的角色和物体。</td></tr>
<tr><td width="56"><img src="https://github.com/PxTicks.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/PxTicks/h3-motion">H3 Motion</a></td><td>编辑主体的关键帧运动，并对遮罩区域重新生成。</td></tr>
<tr><td width="56"><img src="https://github.com/zxcHolmes.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zxcHolmes/minimax-h3-studio">MiniMax-H3 Studio</a></td><td>可自托管的网页和 API 工作室，带持久队列，支持 FastH3 和 Pika H3 Flex。</td></tr>
<tr><td width="56"><img src="https://github.com/Klimentios.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Klimentios/talemondo-studio">TaleMondo Studio</a></td><td>非商用的 Mac 影视制作工作台，用本地 H3 生成。</td></tr>
<tr><td width="56"><img src="https://github.com/thezolon.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/thezolon/tiiny-studio">Tiiny Studio</a></td><td>Tiiny Pocket 硬件的桌面应用，支持 H3-Turbo 文生视频和图生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/slogonomo.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/slogonomo/lf-studio">LF Studio</a></td><td>NVIDIA Sol-H3 的网页工作室，跑在一台 DGX Spark 或一张 RTX 5090 上，原名 sol-h3-studio。</td></tr>
<tr><td width="56"><img src="https://github.com/omnichar.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/omnichar/OmniChar">Omnichar Studio</a></td><td>角色工作室，让同一角色的脸、身材、服装和声音在各个镜头里保持一致；可用 H3 生成，也能在本地训练 H3 LoRA。原名 Inline Studio。</td></tr>
<tr><td width="56"><img src="https://github.com/deepbeepmeep.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/deepbeepmeep/WanGP">WanGP</a></td><td>面向低显存显卡的视频生成工具，支持的模型里有 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/mcmonkeyprojects.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/mcmonkeyprojects/SwarmUI">SwarmUI</a></td><td>网页界面，支持的视频模型里有 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/tin2tin.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tin2tin/Pallaidium">Pallaidium</a></td><td>嵌在 Blender 视频编辑器里的生成式影视工具，支持 H3 和 Open-H3 Regenerate 放大。</td></tr>
<tr><td width="56"><img src="https://github.com/mrbizarro.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/mrbizarro/Phosphene">Phosphene</a></td><td>原生 Mac 应用，在苹果芯片上离线运行 H3 和 LTX-2.5；Pro 版收费，可选。</td></tr>
<tr><td width="56"><img src="https://github.com/neuregex.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/neuregex/freefield">freefield</a></td><td>跑在 Modal 上的一组 H3 应用：文字或照片生成视频、续写片段、换机位。</td></tr>
<tr><td width="56"><img src="https://github.com/AbrahamPaulJ.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/AbrahamPaulJ/nightmare-mobile">nightmare-mobile</a></td><td>安卓上的节点图应用，在骁龙手机上生成 H3 有声视频；非商用许可。</td></tr>
<tr><td width="56"><img src="https://github.com/SwotAtmk.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/SwotAtmk/infinite-creation">infinite-creation</a></td><td>Agent 应用，用本地 ComfyUI 的 H3 把小说或剧本做成任意时长的视频；需要 12 GB 以上显存。</td></tr>
<tr><td width="56"><img src="https://github.com/saddam213.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/saddam213/AmuseAI">Amuse</a></td><td>基于 .NET 的桌面应用，在本地生成图片、视频和音频，H3 是它支持的视频模型之一，可用 CUDA 或 Vulkan。</td></tr>
<tr><td width="56"><img src="https://github.com/james-see.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/james-see/ltx-video-mac">ltx-video-mac</a></td><td>macOS 原生应用，通过 h3.c 在苹果芯片上运行 H3，可选 BF16、INT8 或 6 步 Turbo 版本，附本地 REST API。</td></tr>
<tr><td width="56"><img src="https://github.com/code-video-model.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/code-video-model/CodeVideoModel">CodeVideoModel</a></td><td>先让大模型搭 Three.js 场景渲染成布局视频，再用本地 H3 Ref2VA 生成写实视频。</td></tr>
<tr><td width="56"><img src="https://github.com/OpenDCAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/OpenDCAI/GameFactory-3A">GameFactory-3A</a></td><td>游戏资产生成框架，其中 CG 视频环节用本地 H3 以 720p 生成过场动画和宣传片。</td></tr>
<tr><td width="56"><img src="https://github.com/siyuan-liu31.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/siyuan-liu31/minimax-h3-video-studio">MiniMax H3 Video Studio</a></td><td>基于 ComfyUI 的自托管工作室和命令行工具，支持 H3 文生、图生、参考生和视频生视频，以及分段拼接的长视频。</td></tr>
<tr><td width="56"><img src="https://github.com/RachelForster.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/RachelForster/Shinsekai">Shinsekai</a></td><td>AI 角色扮演游戏制作工具，可通过本地 ComfyUI 用 H3 图生视频把角色立绘做成动态立绘。</td></tr>
<tr><td width="56"><img src="https://github.com/jnMetaCode.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/jnMetaCode/openshorts">OpenShorts</a></td><td>本地优先的短视频流水线，AI 短剧模式可通过 stable-diffusion.cpp 在本机用 H3 生成片段，也可走云端。</td></tr>
</table>

<a id="h3-api"></a>

### API 集成

<table>
<tr><td width="56"><img src="https://github.com/NousResearch.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NousResearch/hermes-agent">Hermes Agent</a></td><td>会自我改进的 AI Agent，自带的 fal 视频插件可调用 H3、H3 Max 和 H3 Max Turbo，并把常用分辨率换算成各自的档位。</td></tr>
<tr><td width="56"><img src="https://github.com/n8n-io.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/n8n-io/n8n/tree/master/packages/@n8n/nodes-langchain/nodes/vendors/MiniMax">n8n</a></td><td>工作流自动化工具，MiniMax 节点可直接调用官方 API 生成 H3 视频；采用 fair-code 许可。</td></tr>
<tr><td width="56"><img src="https://github.com/harry0703.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/harry0703/MoneyPrinterTurbo">MoneyPrinterTurbo</a></td><td>根据主题或关键词生成短视频，可通过秘塔的 API 用 H3 生成视频素材。</td></tr>
<tr><td width="56"><img src="https://github.com/OpenMontageInc.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/OpenMontageInc/OpenMontage">OpenMontage</a></td><td>Agent 驱动的视频制作系统，内置 H3 技能和工具，可调用 MiniMax 官方 API、fal 或本地 ComfyUI。</td></tr>
<tr><td width="56"><img src="https://github.com/BerriAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BerriAI/litellm">LiteLLM</a></td><td>Python SDK 和 AI 网关，可用 OpenAI 风格的视频接口在 MiniMax 官方 API 上创建 H3 任务。</td></tr>
<tr><td width="56"><img src="https://github.com/QuantumNous.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/QuantumNous/new-api">New API</a></td><td>可自托管的 AI 网关，海螺任务插件会把 H3 多模态任务转发到 MiniMax 的 v2 视频接口。</td></tr>
<tr><td width="56"><img src="https://github.com/HeyPuter.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/HeyPuter/puter">Puter</a></td><td>可自托管的网页桌面系统，视频生成的 MiniMax 供应商默认用 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/THU-MAIC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/THU-MAIC/OpenMAIC">OpenMAIC</a></td><td>多 Agent 互动课堂，视频适配器走 MiniMax 的 v2 接口调用 H3 和 H3 Max，可直连，也可经 TokenDance 网关。</td></tr>
<tr><td width="56"><img src="https://github.com/agno-agi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/agno-agi/agno">Agno</a></td><td>Agent 框架，自带的 MiniMax 工具走官方 API，用 H3 根据文字生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/storytold.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/storytold/artcraft">ArtCraft</a></td><td>桌面应用，先在 2D 和 3D 里布置场景，再通过自家云服务或 fal 用 H3 生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/agentscope-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/agentscope-ai/QwenPaw">QwenPaw</a></td><td>个人 AI 助手，Creator 插件可走 MiniMax 官方 API 调用 H3 或 H3 Max，也能接自己在 SGLang 上部署的 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/yetone.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/yetone/magpie">Magpie</a></td><td>菜单栏应用，为各种编程 Agent 统一配置模型；填入 MiniMax API Key 后还能用 H3 生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/simstudioai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/simstudioai/sim">Sim</a></td><td>搭建 Agent 工作流的平台，视频工具可走 MiniMax 官方 API 调用 H3 及其 Highspeed、Lite 和 Max 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/vercel.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vercel/ai">AI SDK</a></td><td>Vercel 的 TypeScript 工具包，MiniMax provider 可生成 H3 和 H3 Max 视频，并按模式检查输入是否合规。</td></tr>
<tr><td width="56"><img src="https://github.com/1Panel-dev.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/1Panel-dev/MaxKB">MaxKB</a></td><td>企业级智能体平台，在 MiniMax 模型供应商下提供 H3 和 H3 Max 的文生视频与图生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/HBAI-Ltd.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/HBAI-Ltd/Toonflow-app">Toonflow</a></td><td>从剧本、分镜到成片的短剧制作工具，通过秘塔、优云智算或 APIMart 的 API 调用 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/chatfire-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/chatfire-AI/huobao-drama">Huobao Drama</a></td><td>一句话生成完整短剧的平台，内置调用 MiniMax 官方 API 的 H3 适配器；非商用许可。</td></tr>
<tr><td width="56"><img src="https://github.com/corsairdev.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/corsairdev/corsair">Corsair</a></td><td>应用集成平台，MiniMax 插件让 Agent 能通过官方 API 生成 H3 和 H3 Max 视频。</td></tr>
<tr><td width="56"><img src="https://github.com/waooAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/waooAI/waoowaoo">waoowaoo</a></td><td>影视全流程生产平台，通过 OpenRouter 调用 H3 和 H3 Max；采用 Elastic License 2.0。</td></tr>
<tr><td width="56"><img src="https://github.com/maximhq.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/maximhq/bifrost">Bifrost</a></td><td>AI 网关，把 H3 和 H3 Max 作为异步任务发到 MiniMax 的 v2 接口，并把尺寸和时长调整到各模型的范围内。</td></tr>
<tr><td width="56"><img src="https://github.com/rikkahub.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/rikkahub/rikkahub">RikkaHub</a></td><td>支持多家大模型的安卓聊天应用，也能通过 MiniMax 官方 API 用 H3 生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/TeamWiseFlow.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/TeamWiseFlow/xiaobei">xiaobei</a></td><td>面向中小企业的自媒体获客 Agent，视频技能走 MiniMax 官方 API 用 H3 生成片段；在 MIT 基础上附加了条款。</td></tr>
<tr><td width="56"><img src="https://github.com/openJiuwen-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/openJiuwen-ai/jiuwenswarm">JiuwenSwarm</a></td><td>AI Agent，视频工具可把 H3 和 H3 Max 任务发到 MiniMax 官方 API。</td></tr>
<tr><td width="56"><img src="https://github.com/Devin-AXIS.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Devin-AXIS/iPolloWork">iPolloWork</a></td><td>Agent 工作台，带 H3 视频和 H3 数字人两种生成工具，都在 RunningHub 工作流上运行；仅限个人非商用。</td></tr>
<tr><td width="56"><img src="https://github.com/ArcReel.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ArcReel/ArcReel">ArcReel</a></td><td>把小说和剧本做成分镜和视频的 Agent 工作台，内置调用 MiniMax 官方 API 的 H3 接口。</td></tr>
<tr><td width="56"><img src="https://github.com/monid-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/monid-ai/monid">Monid</a></td><td>面向 Agent 的工具 API 市场，提供对接 MiniMax v2 接口的 H3、H3 Max 和 H3 Max Turbo。</td></tr>
<tr><td width="56"><img src="https://github.com/EverMind-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/EverMind-AI/Raven">Raven</a></td><td>多 Agent 系统，自带的 MiniMax 媒体工具通过 v2 接口生成 H3 和 H3 Max 视频。</td></tr>
<tr><td width="56"><img src="https://github.com/pollinations.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/pollinations/pollinations">Pollinations</a></td><td>开源生成式 AI 平台，可通过它的 API 生成 H3、H3 Max 和 H3 Max Turbo 视频，上游是 fal。</td></tr>
<tr><td width="56"><img src="https://github.com/chaitin.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/chaitin/MonkeyCode">MonkeyCode</a></td><td>企业级 AI 办公工作台，视频服务走 MiniMax 官方 API，支持 H3 的文生、首尾帧和参考图模式。</td></tr>
<tr><td width="56"><img src="https://github.com/darkzOGx.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/darkzOGx/youtube-automation-agent">YouTube Automation Agent</a></td><td>全流程运营 YouTube 频道的 Agent，可通过 MiniMax 官方 API 用 H3 生成片段，并限制付费秒数。</td></tr>
<tr><td width="56"><img src="https://github.com/crmne.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/crmne/ruby_llm">RubyLLM</a></td><td>Ruby AI 框架，接入 MiniMax 视频协议后，<code>animate</code> 调用可以生成 H3 视频。</td></tr>
<tr><td width="56"><img src="https://github.com/Agent-Field.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Agent-Field/agentfield">AgentField</a></td><td>把 Agent 当作服务来构建和运行的框架，MiniMax 媒体供应商默认用 H3 生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/jnMetaCode.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/jnMetaCode/agency-orchestrator">Agency Orchestrator</a></td><td>多 Agent 工作流工具，视频步骤可通过秘塔或 APIMart 的 API 用 H3 生成片段。</td></tr>
<tr><td width="56"><img src="https://github.com/xuanyustudio.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xuanyustudio/LocalMiniDrama">LocalMiniDrama</a></td><td>数据不出本机的 AI 短剧工具，可通过 MiniMax 官方 API 或中转服务用 H3 生成镜头。</td></tr>
<tr><td width="56"><img src="https://github.com/glanderness.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/glanderness/BeefTV">BeefTV</a></td><td>画布式视频工作台，可在 fal 上实时使用 H3 Max Director，另有对接 MiniMax v2 接口的 H3 插件；源码可见许可。</td></tr>
<tr><td width="56"><img src="https://github.com/WhiteTowerAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/WhiteTowerAI/ohmygame">OhMyGame</a></td><td>用 AI 做游戏的工作室应用，视频工具走 MiniMax 官方 API，可选 H3 和 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/ddcat-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ddcat-ai/open-ai-canvas">open-ai-canvas</a></td><td>AI 影视创作画布，内置 H3 提示词卡片，可通过 MiniMax、阿里云百炼或秘塔调用 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/Stonewuu.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Stonewuu/ai-fusion-video">ai-fusion-video</a></td><td>基于 Java 的短剧漫剧 Agent 平台，内置调用 MiniMax 官方 API 的 H3 和 H3 Max 预设。</td></tr>
<tr><td width="56"><img src="https://github.com/siddsachar.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/siddsachar/row-bot">Row-Bot</a></td><td>本地优先的个人助手，视频工具可走 MiniMax 官方 API 调用 H3 及其 Highspeed、Lite 和 Max 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/869413421.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/869413421/ai-moive-studio">AICON</a></td><td>以无限画布驱动的视频创作 Agent，默认视频模型是 H3，通过项目自家的 API 网关调用。</td></tr>
<tr><td width="56"><img src="https://github.com/tigerowo.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tigerowo/infinite-canvas">infinite-canvas</a></td><td>AI 创作画布，可按 MiniMax 接口协议生成 H3 视频，内置 H3 提示词技能。</td></tr>
<tr><td width="56"><img src="https://github.com/xixihhhh.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xixihhhh/clipforge">ClipForge</a></td><td>上传一张商品图生成带货短视频，可通过 Atlas Cloud 调用 H3 和 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/0xsline.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/0xsline/StoryGen-Atelier">StoryGen-Atelier</a></td><td>分镜工具，由 Gemini 写分镜和画面，视频片段除 Veo 外也可通过 MiniMax 官方 API 用 H3 生成。</td></tr>
<tr><td width="56"><img src="https://github.com/looplj.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/looplj/reveric">Reveric</a></td><td>小说和短剧桌面创作工作台，可通过 MiniMax 或秘塔的 API 用 H3 生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/anymouschina.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/anymouschina/TapCanvas">TapCanvas</a></td><td>面向 AI 影视和漫剧的 Agent 画布，自带的网关支持 MiniMax 的 H3 v2 视频协议。</td></tr>
<tr><td width="56"><img src="https://github.com/ChrisChen667788.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ChrisChen667788/wind-comic">wind-comic</a></td><td>多 Agent 流水线，把一句话做成短剧，镜头通过 MiniMax 官方 API 用 H3 生成。</td></tr>
<tr><td width="56"><img src="https://github.com/agentjido.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/agentjido/req_llm">ReqLLM</a></td><td>Elixir 大模型库，<code>generate_video</code> 可调用 MiniMax 官方 API 的 H3，也能直接上传本地参考文件。</td></tr>
<tr><td width="56"><img src="https://github.com/ZMGID.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ZMGID/kivio">Kivio</a></td><td>macOS 和 Windows 上的屏幕级 AI 助手，可通过 MiniMax v2 接口生成 H3 视频。</td></tr>
<tr><td width="56"><img src="https://github.com/tryonlabs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tryonlabs/opentryon">OpenTryOn</a></td><td>虚拟试衣和时尚 AI 工具包，可通过 MiniMax 官方 API 调用 H3 和 H3 Max，也能在本地显卡上跑 H3；非商用许可。</td></tr>
<tr><td width="56"><img src="https://github.com/ganbo-gab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ganbo-gab/open-storyboard-canvas">Open Storyboard Canvas</a></td><td>AI 分镜和导演台画布，视频节点可使用预设的 MiniMax H3 供应商。</td></tr>
<tr><td width="56"><img src="https://github.com/fancyboi999.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/fancyboi999/Loomic">Loomic</a></td><td>在无限画布上用对话做设计的 Agent，可通过秘塔的 API 用 H3 从文字或首尾帧生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/chillzhuang.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/chillzhuang/Kinema">Kinema</a></td><td>做 AI 漫剧、电影和解说视频的 Agent，带对接 MiniMax v2 接口的 <code>minimax-h3</code> 供应商。</td></tr>
<tr><td width="56"><img src="https://github.com/Finderchangchang.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Finderchangchang/brewreel">BrewReel</a></td><td>根据产品简报生成竖版宣传片；口播配画面模式可通过 MiniMax 官方 API 用 H3 生成 AI 画面。</td></tr>
<tr><td width="56"><img src="https://github.com/LudwigKienle.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/LudwigKienle/ai-video-production-editor">AI Video Production Editor</a></td><td>面向 AI 电影制作的 Electron 编辑器，节点流程里可通过 fal 运行 H3 文生视频和图生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/sumarilkkxx.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/sumarilkkxx/Mora">Mora</a></td><td>本地优先的商业推广视频工作台，可通过 Atlas Cloud 用 H3 生成视频片段。</td></tr>
<tr><td width="56"><img src="https://github.com/saihhold-zhao.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/saihhold-zhao/polox_ai">Polox AI</a></td><td>基于 DeepSeek Harness 的多模态生成平台，可通过 WaveSpeed 调用 H3 文生视频、图生视频和参考生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/s-nagaev.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/s-nagaev/chibi">Chibi</a></td><td>可自托管的 Telegram 机器人，视频生成可通过 MiniMax 官方 API 调用 H3 和 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/WaveSpeedAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/WaveSpeedAI/mcp-server">WaveSpeed MCP Server</a></td><td>MCP 服务，可在 WaveSpeed 的 API 上用 H3 生成、编辑和延长视频。</td></tr>
<tr><td width="56"><img src="https://github.com/Sociaro-Research.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Sociaro-Research/open-media-gateway">Open Media Gateway</a></td><td>可自托管的媒体服务，MiniMax 供应商覆盖官方 API 上的 H3 及其 Highspeed、Lite 和 Max 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/KunLabAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/KunLabAI/KunFlix">KunFlix</a></td><td>影视广告创作平台，视频服务通过 MiniMax v2 接口调用 H3；Apache-2.0 附加条款。</td></tr>
<tr><td width="56"><img src="https://github.com/Jjat00.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Jjat00/hf-studio">hf-studio</a></td><td>可自托管的媒体工作室，通过 Higgsfield 调用 H3 视频，通过 KIE 调用 H3-Image。</td></tr>
<tr><td width="56"><img src="https://github.com/ItusiAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ItusiAI/Open-Magiviz">Magiviz</a></td><td>AI 视频创作引擎，视频模型里有通过 KIE API 调用的 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/openfilm.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/openfilm/openfilm">OpenFilm</a></td><td>让编程 Agent 做视频的工具和编辑器，可通过 fal 的 H3 Max Recast 替换片段里的人物。</td></tr>
<tr><td width="56"><img src="https://github.com/HaD0Yun.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/HaD0Yun/CozyClay">CozyClay</a></td><td>浏览器里的预演工具，先搭场景和机位，再用 fal 上的 H3 Max Turbo 或自己的 ComfyUI H3 工作流渲染镜头。</td></tr>
<tr><td width="56"><img src="https://github.com/openstory-so.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/openstory-so/openstory">OpenStory</a></td><td>视频序列创作平台，运动生成环节可通过 fal 使用 H3 Max 文生视频、图生视频和参考生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/lovisdotio.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/lovisdotio/video-to-game">video-to-game</a></td><td>31 款游戏，每款都由 fal 上生成的 10 到 15 秒 H3 Max 片段复刻而来，附素材制作流程。</td></tr>
<tr><td width="56"><img src="https://github.com/patch-gg.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/patch-gg/anymove">anymove</a></td><td>用 fal 上生成的双机位 H3 Max 片段为任意生物做动作捕捉，附在 fal 上训练 H3 LoRA 的技能。</td></tr>
<tr><td width="56"><img src="https://github.com/blendi-remade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/blendi-remade/prompt2rig">prompt2rig</a></td><td>输入一个动作，绑好骨骼的角色就会做出来，快速模式用 fal 上的 H3 Max Turbo 生成视频。</td></tr>
<tr><td width="56"><img src="https://github.com/blendi-remade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/blendi-remade/falcraft">falcraft</a></td><td>Minecraft 模组，用 fal 生成建筑，其中墙上的电视播放由 fal 上的 H3 Max Director 实时生成的画面。</td></tr>
<tr><td width="56"><img src="https://github.com/vargHQ.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vargHQ/sdk">varg SDK</a></td><td>TypeScript 视频生成 SDK，内置 fal 上的 H3 Max 文生视频、图生视频和参考生视频模型；JSX 写法已弃用。</td></tr>
<tr><td width="56"><img src="https://github.com/orkunisitmak.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/orkunisitmak/banana-studio">Banana Studio</a></td><td>可自托管的媒体工作室，可通过 fal 使用 H3 Max 的文生视频、图生视频、参考生视频、延长、对口型和换人。</td></tr>
<tr><td width="56"><img src="https://github.com/henjicc.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/henjicc/Henji-AI">Henji AI</a></td><td>聚合多家供应商的桌面应用，可通过 fal 或 APIMart 调用 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/xflare-bot.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xflare-bot/pokemonlive">pokemonlive</a></td><td>本地宝可梦对战原型，过场动画由 fal 上的 H3 Max Turbo 和 H3 Max 参考生视频生成。</td></tr>
<tr><td width="56"><img src="https://github.com/internetphysics.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/internetphysics/live-classroom">live-classroom</a></td><td>教任何主题的直播电视频道，每段课程内容都由 fal 上的 H3 Max Turbo 即时生成。</td></tr>
<tr><td width="56"><img src="https://github.com/gokayfem.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gokayfem/h3-max-education">H3 Max Realtime Education</a></td><td>可随时用语音打断的实时科学辅导，教学视频由 fal 上的 H3 Max 持续生成。</td></tr>
<tr><td width="56"><img src="https://github.com/marketcalls.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/marketcalls/sketchmaker">SketchMaker</a></td><td>可自托管的创作工具，动画、故事和对口型功能默认用 fal 上的 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/gokayfem.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gokayfem/H3-Max-Blender">H3 Max Blender</a></td><td>Blender 神经渲染演示，用 fal 上的 H3 Max 把一艘灰模飞船渲染成四种风格，随几何体变化更新。</td></tr>
<tr><td width="56"><img src="https://github.com/Valiera00.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Valiera00/SPITE">SPITE</a></td><td>面向电影人的前期制作画布，视频节点可用你自己的 fal 密钥调用 H3 Max。</td></tr>
<tr><td width="56"><img src="https://github.com/blendi-remade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/blendi-remade/interdimensional-game">Interdimensional Game</a></td><td>实时生成视频的互动冒险游戏，视频由 fal 上的 H3 Max 生成。</td></tr>
<tr><td width="56"><img src="https://github.com/blendi-remade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/blendi-remade/unreel">UNREEL</a></td><td>AI 流媒体应用，大模型写剧集，再由 fal 上的 H3 Max Turbo 逐个镜头生成画面。</td></tr>
<tr><td width="56"><img src="https://github.com/ilkerzg.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ilkerzg/storycast">StoryCast</a></td><td>输入主题生成故事视频，镜头和对口型的台词由 fal 上的 H3 Max 生成。</td></tr>
<tr><td width="56"><img src="https://github.com/eren23.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/eren23/openflipbook">OpenFlipbook</a></td><td>每页都是 AI 插画的翻页书，可选的页面动画由 fal 上的 H3 Max 或 H3 Max Turbo 生成。</td></tr>
</table>

<a id="h3-comfyui"></a>

### ComfyUI 节点

<table>
<tr><td width="56"><img src="https://github.com/T8mars.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/T8mars/comfyui-minimax-h3-audio-T8">comfyui-minimax-h3-audio-T8</a></td><td>62 个节点，覆盖音频、静态图、条件控制、长视频和语音。</td></tr>
<tr><td width="56"><img src="https://github.com/huangserva.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/huangserva/ComfyUI_MiniMaxH3_Director">ComfyUI_MiniMaxH3_Director (huangserva)</a></td><td>分镜节点，附五个可导入模板：t2v、fl2v、r2v、v2v 和 rv2v。</td></tr>
<tr><td width="56"><img src="https://github.com/AIMixer.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director">ComfyUI_MiniMaxH3_Director (AIMixer)</a></td><td>最早的 Director 节点。</td></tr>
<tr><td width="56"><img src="https://github.com/seesee75-commits.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/seesee75-commits/ComfyUI-MiniMaxH3-Director">ComfyUI-MiniMaxH3-Director</a></td><td>时间线编辑器：素材拖上轨道、每个镜头一条提示词，支持实时预览和重拍。</td></tr>
<tr><td width="56"><img src="https://github.com/imbutus.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/imbutus/ComfyUI-MiniMaxDirector">ComfyUI-MiniMaxDirector</a></td><td>把镜头、运镜和音频提示排在时间线上，编译成官方指南里的结构化提示词。</td></tr>
<tr><td width="56"><img src="https://github.com/nkxx188.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy">ComfyUI-MiniMaxH3-Easy</a></td><td>一个紧凑的工作流覆盖文生视频、图生视频、首尾帧和参考生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/FX-FeiHou.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/FX-FeiHou/ComfyUI-FeiHou-Easy-H3">ComfyUI-FeiHou-Easy-H3</a></td><td>MiniMaxH3-Easy 的分支版，内置 9 图、3 视频、3 音频的参考槽位。</td></tr>
<tr><td width="56"><img src="https://github.com/NikoDemon80.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context">ComfyUI-H3-Motion-Context</a></td><td>把多段视频接起来，保持动作和声音连贯。</td></tr>
<tr><td width="56"><img src="https://github.com/princepainter.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/princepainter/ComfyUI-PainterNodes">ComfyUI-PainterNodes</a></td><td><code>MiniMaxRefToVideo2</code> 节点，支持官方的参考和对白格式。</td></tr>
<tr><td width="56"><img src="https://github.com/wyzborrero.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/wyzborrero/ComfyUI-H3-FunControl">ComfyUI-H3-FunControl</a></td><td>加载 Fun ControlNet-Union，附预处理器和局部重绘流程。</td></tr>
<tr><td width="56"><img src="https://github.com/lihaoyun6.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/lihaoyun6/ComfyUI-MiniMaxH3_Ref-Patch">ComfyUI-MiniMaxH3_Ref-Patch</a></td><td>把 Ref Patch 应用到 FL2VA 上。</td></tr>
<tr><td width="56"><img src="https://github.com/zechengtang.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zechengtang/VPack">VPack</a></td><td>压缩 Ref2VA 的参考视频条件，减少参考 token 数量。</td></tr>
<tr><td width="56"><img src="https://github.com/alisson-anjos.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/alisson-anjos/ai-toolkit/tree/tara-h3-v0.1.2/research/tara-h3">TARA-H3</a></td><td>实验性的 Ref2VA 视频参考注意力时间对齐，附技术报告。</td></tr>
<tr><td width="56"><img src="https://github.com/matlowai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/matlowai/ComfyUI-Dials">ComfyUI-Dials</a></td><td>用提示词对构造免训练的引导向量，调节动作和风格。</td></tr>
<tr><td width="56"><img src="https://github.com/scraed.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/scraed/LanPaint">LanPaint</a></td><td>免训练的视频和音频局部重绘。</td></tr>
<tr><td width="56"><img src="https://github.com/Tr1dae.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler">ComfyUI-MiniMaxH3_LatentUpscaler</a></td><td>针对 H3 音视频潜变量的放大节点，用于两段式采样。</td></tr>
<tr><td width="56"><img src="https://github.com/LBH-123-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler">Comfyui_Minimax_h3_latent_Upscaler</a></td><td>学习型潜空间放大，1.0 到 4.0 倍，不必先解码再编码。</td></tr>
<tr><td width="56"><img src="https://github.com/tazmannner.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tazmannner/ComfyUI-H3-ResCascade">ComfyUI-H3-ResCascade</a></td><td>高清修复节点，把约 100 万像素的结果精修到 1080p 或 1440p，保留原音频。</td></tr>
<tr><td width="56"><img src="https://github.com/rdh9616.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate">ComfyUI Open-H3 Regenerate</a></td><td>NVIDIA Open-H3-Regenerate-2K 精修模型的节点和工作流。</td></tr>
<tr><td width="56"><img src="https://github.com/hashu786.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/hashu786/ComfyUI-Tessera">Tessera</a></td><td>分块、分段的视频精修，VAE 编解码以流式进行。</td></tr>
<tr><td width="56"><img src="https://github.com/lihaoyun6.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT">ComfyUI-H3VAE_TRT</a></td><td>TensorRT 版视频 VAE，解码最多快 1.7 倍。</td></tr>
<tr><td width="56"><img src="https://github.com/xmarre.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3">ComfyUI-Spectrum-MiniMax-H3</a></td><td>通过预测特征跳过部分 transformer 计算，结果是近似的。</td></tr>
<tr><td width="56"><img src="https://github.com/kijai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/kijai/ComfyUI-SolAttn_triton">ComfyUI-SolAttn_triton</a></td><td>免训练的 Sol-Attn 稀疏注意力，在 H3 上测试过；ComfyUI 已内置这项功能，此节点已停止维护。</td></tr>
<tr><td width="56"><img src="https://github.com/Saganaki22.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Saganaki22/ComfyUI-VDN-H3">ComfyUI-VDN-H3</a></td><td>在 ComfyUI 里运行 VDN 适配器；<a href="https://github.com/Speach1sdef178/ComfyUI-VDN-H3-24GB">24 GB 分支版</a>卸载得更多。</td></tr>
<tr><td width="56"><img src="https://github.com/Kablex.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Kablex/ComfyUI-Ref2VA-VSA">ComfyUI-Ref2VA-VSA</a></td><td>给 Ref2VA 用的视频稀疏注意力，RTX 4090 上 5 秒视频约 72 秒。</td></tr>
<tr><td width="56"><img src="https://github.com/Jalen-Brunson.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc">ComfyUI-MiniMax-H3-PDD-Acc</a></td><td>在旧版 ComfyUI 上加载带 32 个输出头的 PDD LoRA。</td></tr>
<tr><td width="56"><img src="https://github.com/xmarre.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/xmarre/ComfyUI-Sol-H3">ComfyUI-Sol-H3</a></td><td>Sol-H3 运行时的 ComfyUI 前端（GPL-3.0）。</td></tr>
<tr><td width="56"><img src="https://github.com/zechengtang.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zechengtang/Spark-H3">Spark-H3</a></td><td>自适应块稀疏注意力，提供 diffusers 和 ComfyUI 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/iamwavecut.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/iamwavecut/ComfyUI-OrbitQuant/tree/feature/minimax-h3-comfyui">ComfyUI-OrbitQuant</a></td><td>加载 OrbitQuant W4A4 版本。</td></tr>
<tr><td width="56"><img src="https://github.com/NikoDemon80.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/NikoDemon80/ComfyUI-FastH3-Lora-Converter">ComfyUI-FastH3-Lora-Converter</a></td><td>把 FastH3 LoRA 转成 ComfyUI 的键名顺序。</td></tr>
<tr><td width="56"><img src="https://github.com/BobJohnson24.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BobJohnson24/ComfyUI-INT8-Fast">ComfyUI-INT8-Fast</a></td><td>已基本被原生 INT8 取代，<code>convert_comfy_quant.py</code> 仍可转换旧版 INT8 文件。</td></tr>
<tr><td width="56"><img src="https://github.com/Songssx.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector">ComfyUI-MiniMaxH3-TimelineDirector</a></td><td>可编辑的 Ref2VA 参考素材时间线，附一个全功能导演台工作流。</td></tr>
<tr><td width="56"><img src="https://github.com/Luisacaotica.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod">ComfyUI-MiniMaxH3Mod</a></td><td>RefMod 适配器，不用训练就把参考图和视频压成很小的潜变量；仍在开发中。</td></tr>
<tr><td width="56"><img src="https://github.com/Carasibana.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Carasibana/ComfyUI-H3-FaceRefine">ComfyUI-H3-FaceRefine</a></td><td>逐帧跟踪画面里的小脸，用 H3 精修后再贴回去。</td></tr>
<tr><td width="56"><img src="https://github.com/JGRFW.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/JGRFW/comfyui-AICG3D">comfyui-AICG3D</a></td><td>短剧工作台，把剧本拆成多段，生成任意时长的视频。</td></tr>
<tr><td width="56"><img src="https://github.com/ukr8b3g-cmyk.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ukr8b3g-cmyk/H3-Character-Sheet-Designer">H3-Character-Sheet-Designer</a></td><td>用一张角色图生成角色设定图，最多 13 个视角。</td></tr>
<tr><td width="56"><img src="https://github.com/tritant.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tritant/ComfyUI_MiniMax_H3_Extender">ComfyUI_MiniMax_H3_Extender</a></td><td>把多段片段接成长视频，动作、画面和声音保持连贯。</td></tr>
<tr><td width="56"><img src="https://github.com/MajoorWaldi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MajoorWaldi/ComfyUI-Majoor-H3-GuideMaster">ComfyUI-Majoor-H3-GuideMaster</a></td><td>把图片和音频放到 H3 时间线上作为引导。</td></tr>
<tr><td width="56"><img src="https://github.com/Saganaki22.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3">ComfyUI-Viggle-Animate-H3</a></td><td>Viggle Animate 的节点，用参考图替换驱动视频里的角色。</td></tr>
<tr><td width="56"><img src="https://github.com/astropuzzo.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio">ComfyUI-MiniMax-H3-Image-Studio</a></td><td>用视频模型生成静态图和做参考图编辑，附 Qwen 细节精修。</td></tr>
<tr><td width="56"><img src="https://github.com/thaakeno.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/thaakeno/ComfyUI-MiniMax-H3-Studio">ComfyUI-MiniMax-H3-Studio</a></td><td>基于视频模型、支持参考图的静态图工作流，目前是 alpha 版。</td></tr>
<tr><td width="56"><img src="https://github.com/HELPMEEADICE.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS">TE-Speed-MiniMaxH3-OSS</a></td><td>步间缓存，sigma 变化很小时复用后段层的残差，30 步时约快 45%；需要给 ComfyUI 打一次补丁。</td></tr>
<tr><td width="56"><img src="https://github.com/vizart-vj.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia">ComfyUI-MiniMax-H3-LongMedia</a></td><td>长篇音视频节点，带 Director 时间线，可单独重渲某一段，也能重配口型。</td></tr>
<tr><td width="56"><img src="https://github.com/ukr8b3g-cmyk.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum">ComfyUI-H3-Continuum</a></td><td>分段生成长视频和音频，每段一条提示词，可单独重试某一段，也能从保存的结果接着生成。</td></tr>
<tr><td width="56"><img src="https://github.com/seitanism.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef">ComfyUI-H3-Motion-Context-MultiRef</a></td><td>H3-Motion-Context 的分支版，新增围绕一首歌生成的音乐视频、V2V 动作迁移、两段视频之间的音视频过渡和带遮罩的关键帧。</td></tr>
<tr><td width="56"><img src="https://github.com/chanon.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/chanon/comfyui-obvpm-timeline">comfyui-obvpm-timeline</a></td><td>时间线节点，像小型剪辑软件一样生成、延长、往前补、衔接片段，也能做无缝循环；需要 comfyui-obvpm 节点包。</td></tr>
<tr><td width="56"><img src="https://github.com/yichengup.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3">ComfyUI-YCNodes-MiniMax-H3</a></td><td>6 个节点，包括按时间分段的提示词、动态 CFG、低噪阶段精修和分块采样。</td></tr>
<tr><td width="56"><img src="https://github.com/goohai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/goohai/Goohai-MiniMax-H3_Integration">Goohai-MiniMax-H3_Integration</a></td><td>一体化条件节点，按上传的素材自动选择 T2VA、FL2VA、Ref2VA 或混合模式，并写好官方标签。</td></tr>
<tr><td width="56"><img src="https://github.com/bbaudio-2025.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale">Comfyui-MMH3-UltimateUpscale</a></td><td>把生成好的潜变量放大后按时间分段、按空间分块重新采样，让高分辨率长视频在有限显存里跑完。</td></tr>
<tr><td width="56"><img src="https://github.com/matlowai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/matlowai/ComfyUI-MAINodes">ComfyUI-MAINodes</a></td><td>Motion Lab 先渲染放慢版再去掉重复帧，修复快速动作的拖影；另有多视角和低显存节点。</td></tr>
<tr><td width="56"><img src="https://github.com/knoic.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream">ComfyUI-MiniMaxH3-PrefixStream</a></td><td>长视频续写节点，把上一段干净的潜变量写进下一段开头，并用素材箱挑选从哪一段接着生成。</td></tr>
<tr><td width="56"><img src="https://github.com/nicolab28.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/nicolab28/ComfyUI-ClipProj">ComfyUI-ClipProj</a></td><td>用学习到的<a href="https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3">投影</a>把 32B 文本编码器换成 Qwen3-VL 4B 或 8B，最低只占 4.5 GB。</td></tr>
<tr><td width="56"><img src="https://github.com/scottmudge.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader">ComfyUI_MinimaxH3HybridLoader</a></td><td>加载时按层段把 Ref2VA 的部分权重合入 FL2VA，不用另存新文件。</td></tr>
<tr><td width="56"><img src="https://github.com/shootthesound.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still">ComfyUI-Fizgig-H3-Still</a></td><td>两个节点，用单帧潜变量和干净的解码让 H3 出真正的单帧静态图，尺寸不限，也不需要额外模型。</td></tr>
<tr><td width="56"><img src="https://github.com/Adudeguyman.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Adudeguyman/ComfyUI-H3-AudioRefine">ComfyUI-H3-AudioRefine</a></td><td>冻结已采样潜变量里的视频，只给音频多跑几步，适合画面够用但声音粗糙的 4 步 Turbo 结果。</td></tr>
<tr><td width="56"><img src="https://github.com/KadoBOT.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/KadoBOT/ComfyUI-H3-Continuity-Studio">ComfyUI-H3-Continuity-Studio</a></td><td>多镜头时间线，在片段之间和换场景、换机位后延续上下文，附给编程 agent 用的 MCP 接口。</td></tr>
<tr><td width="56"><img src="https://github.com/Saganaki22.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Saganaki22/ComfyUI-sol-attn">ComfyUI-sol-attn</a></td><td>Sol-Attn 的 Triton 内核，附 H3 专用节点：零拷贝注意力、融合 AdaLN 调制和分块 MLP，支持 RTX 30 系到 Blackwell。</td></tr>
<tr><td width="56"><img src="https://github.com/rookiestar28.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/rookiestar28/MiniMaxH3-Studio">MiniMaxH3-Studio</a></td><td>ComfyUI 侧边栏加 28 个节点，覆盖写提示词、调用官方 H3 工作流、规划长视频和内置剪辑。</td></tr>
<tr><td width="56"><img src="https://github.com/veda-sparse.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/veda-sparse/Veda-on-ComfyUI">Veda-on-ComfyUI</a></td><td>一个节点给原生 H3 加上 Veda 稀疏注意力，在 RTX 5070 上端到端快 2.9 倍，支持 NVIDIA、AMD ROCm 和苹果芯片。</td></tr>
<tr><td width="56"><img src="https://github.com/WASasquatch.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/WASasquatch/was-node-suite-comfyui">WAS Node Suite</a></td><td>通用节点套件，H3 节点能在一个节点里写好多段视频的条件、逐窗口延长，并用 Fun ControlNet 控制每段。</td></tr>
<tr><td width="56"><img src="https://github.com/Astral-xm.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Astral-xm/ComfyUI-MiniMax-ContextIR">ComfyUI-MiniMax-ContextIR</a></td><td>覆盖本地 H3 从提示词到采样的节点包，可选用 MiniMax 的 H3 Context-IR API 改写提示词。</td></tr>
</table>

<a id="h3-workflows"></a>

### 工作流

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65167486c1ec63624f5e45dd/zJlq2UH7TzeLv77AsbFUn.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow">MiniMax-H3-Multishot-Workflow</a></td><td>把多段 FL2VA 和 Ref2VA 视频串成一个序列，衔接处音频对齐。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65dc6589ec3dc9ccf761d936/SxJaAV-JarA1HZj4D1iEV.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/javawock7618/comfy-MiniMax-H3-workflows">comfy-MiniMax-H3-workflows</a></td><td>把低显存加速方案打包成一个可导入的工作流。</td></tr>
<tr><td width="56"><img src="https://github.com/FlowForgeLabAi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/FlowForgeLabAi/MiniMax-H3-8GB-Workflows">MiniMax-H3-8GB-Workflows</a></td><td>面向 8 GB 笔记本的三档工作流合集。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/minimax_fl2v_gguf_workflow.json">FL2V GGUF workflow</a></td><td>在 ComfyUI 里运行 GGUF 量化的 FL2VA。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1661165878439-noauth.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-T2VA.json">OrbitQuant workflows</a></td><td>W4A4 版本的文生音视频工作流，另有 <a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-T2VA-api.json">T2VA</a> 和 <a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/workflows/MiniMax-H3-OrbitQuant-Ref2VA-api.json">Ref2VA</a> 的 API 格式。</td></tr>
<tr><td width="56"><img src="https://github.com/pixelaigc.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/pixelaigc/ComfyUI-MiniMax-H3-Workflow">ComfyUI-MiniMax-H3-Workflow</a></td><td>首尾帧、图生视频和文生视频合在一个工作流里，带 Turbo LoRA 和分辨率总控面板。</td></tr>
</table>

<a id="h3-prompting"></a>

### 提示词与 Agent 技能

<table>
<tr><td width="56"><img src="https://github.com/1038lab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor">ComfyUI-MiniMax-H3-Promptor</a></td><td>生成 H3 提示词，把 <code>&lt;Picture X&gt;</code> 标签直接写进动作描述里。</td></tr>
<tr><td width="56"><img src="https://github.com/ethanfel.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide">ComfyUI-MiniMax-H3-Guide</a></td><td>把身份、关键帧、动作、配音和配乐拆成不同角色，再编译成合规的 H3 提示词。</td></tr>
<tr><td width="56"><img src="https://github.com/ruashots.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ruashots/open-h3-ir">OpenH3-IR</a></td><td>按 MiniMax 公开的 Context-IR 格式写提示词，写完自检修正；可用命令行、HTTP 或 ComfyUI。</td></tr>
<tr><td width="56"><img src="https://github.com/T8mars.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8">comfyui-minimax-h3-prompt-enhancer-T8</a></td><td>在 ComfyUI 里调用服务端扩写提示词。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/680de13385293771bc57400b/qnjz3avU_cKzRZA2aGjuZ.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA">MiniMax-H3-Prompt-Rewriter-LoRA</a></td><td>Qwen3.6-27B 的 LoRA，把短提示词改写成 H3 的三段式结构；<a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B">8B 视觉版</a>还能看关键帧。</td></tr>
<tr><td width="56"><img src="https://github.com/BeatAPI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BeatAPI/awesome-minimax-h3-prompts">awesome-minimax-h3-prompts</a></td><td>按类别整理的提示词合集，附示例视频。</td></tr>
<tr><td width="56"><img src="https://github.com/yangzhou-chaofan.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/yangzhou-chaofan/minimax-h3-1000-prompts">MiniMax H3 1K prompts</a></td><td>精选的 1000 条文生视频提示词，注明作者。</td></tr>
<tr><td width="56"><img src="https://github.com/T8mars.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/T8mars/minimax-h3-prompt-skill-T8">minimax-h3-prompt-skill-T8</a></td><td>案例库，可作为 Agent 技能安装，附桌面查看器。</td></tr>
<tr><td width="56"><img src="https://github.com/joeVenner.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/joeVenner/awesome-minimax-h3">awesome-minimax-h3 (skills)</a></td><td>通过托管 API 把需求一路做成成片的 Agent 技能。</td></tr>
<tr><td width="56"><img src="https://github.com/okaypic.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/okaypic/okaypic-video-skill">okaypic-video-skill</a></td><td>通过 okaypic API 生成带字幕多镜头成片的 Agent 技能。</td></tr>
<tr><td width="56"><img src="https://github.com/SlavaSexton.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/SlavaSexton/ComfyUI-Agent-Kit">ComfyUI-Agent-Kit</a></td><td>让编程 Agent 驱动本地 ComfyUI，内含 H3 技能。</td></tr>
<tr><td width="56"><img src="https://github.com/unknowlei.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/unknowlei/minimax-h3-opencode-skills">minimax-h3-opencode-skills</a></td><td>OpenCode 技能集：导演、多镜头规划、写提示词和审片。</td></tr>
<tr><td width="56"><img src="https://github.com/pytraveler.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI">MiniMax-H3-Prompt-Rewriter-ComfyUI</a></td><td>LightX2V 提示词改写 LoRA 的 ComfyUI 节点，另能给参考素材写描述。</td></tr>
<tr><td width="56"><img src="https://github.com/benjiyaya.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/benjiyaya/ComfyUI-H3-VisionPromptor">ComfyUI-H3-VisionPromptor</a></td><td>用本地视觉模型把想法和参考图扩写成 H3 提示词。</td></tr>
<tr><td width="56"><img src="https://github.com/Rimagination.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Rimagination/h3lite">h3lite</a></td><td>Codex 技能，按电脑配置选择 ComfyUI 方案，在本地生成 H3 视频。</td></tr>
<tr><td width="56"><img src="https://github.com/benjiyaya.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/benjiyaya/Minimax-H3-Prompt-AgentSkill">Minimax-H3-Prompt-AgentSkill</a></td><td>根据你的素材和想法生成格式化的 H3 提示词，交回给你自己运行。</td></tr>
<tr><td width="56"><img src="https://github.com/Adudeguyman.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder">ComfyUI-Fantastic-MiniMaxH3-PromptBuilder</a></td><td>每种模式的可填写提示词模板，边写边按官方指南检查。</td></tr>
<tr><td width="56"><img src="https://github.com/duckyshell.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/duckyshell/ComfyUI-MiniMaxH3-Prompt-Writer">ComfyUI-MiniMaxH3-Prompt-Writer</a></td><td>ComfyUI 里的提示词写作面板，把一段简单描述和参考素材写成官方 H3 格式，可用 Ollama、本地 GGUF 或 API 模型。</td></tr>
<tr><td width="56"><img src="https://github.com/gnipbao.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gnipbao/minimax-h3-video-reverse-skill">minimax-h3-video-reverse-skill</a></td><td>Agent 技能，读一段参考视频，写成 H3 文生视频提示词或逐镜头的图生视频提示词。</td></tr>
<tr><td width="56"><img src="https://github.com/phileiny.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/phileiny/h3-storyboard-skill">h3-storyboard-skill</a></td><td>Claude Code 技能，把剧本拆成 H3 镜头并指导面部表演，每条规则都标明是实测还是推断。</td></tr>
<tr><td width="56"><img src="https://github.com/TFboy1.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/TFboy1/oh-my-minimaxh3-director">oh-my-minimaxh3-director</a></td><td>Agent 技能，把剧本拆成分镜，用本地 ComfyUI 工作流批量生成，再拼成剪映草稿。</td></tr>
<tr><td width="56"><img src="https://github.com/Sac-Y.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Sac-Y/MiniMax-H3-Cloud">MiniMax-H3-Cloud</a></td><td>Codex 插件，在优云智算上租用 GPU 并装好 ComfyUI，跑文生视频、关键帧或参考图生视频，结束后关机。</td></tr>
<tr><td width="56"><img src="https://github.com/jtydhr88.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/jtydhr88/ComfyTV">ComfyTV</a></td><td>ComfyUI 上的画布应用，内置 Agent 自带一个 H3 电影导演技能。</td></tr>
<tr><td width="56"><img src="https://github.com/zenstory-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zenstory-ai/drama-skills">drama-skills</a></td><td>从小说到分镜再到提示词的短剧技能集，H3 模式会绑定参考图和说话人。</td></tr>
<tr><td width="56"><img src="https://github.com/eternityspring.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/eternityspring/shuohao-skills">shuohao-skills</a></td><td>短剧技能集，产出角色、美术、剧本设定和分镜，分镜环节把剪辑点和关键帧对齐到 H3 提示词。</td></tr>
<tr><td width="56"><img src="https://github.com/LoveRain1997.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/LoveRain1997/h3-prompt-journal">h3-prompt-journal</a></td><td>以音乐节拍驱动的舞蹈和多镜头提示词案例，每篇附完整提示词和经验总结。</td></tr>
<tr><td width="56"><img src="https://github.com/SkyNotSilent.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/SkyNotSilent/awesome-MiniMax-H3-cases">awesome-MiniMax-H3-cases</a></td><td>3000 多个可播放的 H3 视频，均注明出处，另有 700 多条完整提示词和本地运行教程。</td></tr>
<tr><td width="56"><img src="https://github.com/BMB12d3.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BMB12d3/minimax-h3-prompt-composer">H3 Prompt Composer</a></td><td>单个 HTML 文件的离线提示词构建工具，直接在浏览器里运行。</td></tr>
<tr><td width="56"><img src="https://github.com/scenario-labs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/scenario-labs/skills">Scenario skills</a></td><td>调用 Scenario API 的 Agent 技能集，其中的视频技能讲解 H3 的关键帧和参考模式以及原生音频提示词。</td></tr>
<tr><td width="56"><img src="https://github.com/ChatCut-Inc.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ChatCut-Inc/agent-plugin">ChatCut Agent Plugin</a></td><td>供编程 Agent 使用的 AI 视频编辑工具，视频技能把 H3 和 H3 Max 任务提交到它的托管服务。</td></tr>
<tr><td width="56"><img src="https://github.com/zenstory-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zenstory-ai/oh-story-dsh">oh-story-dsh</a></td><td>面向网文和短剧的 DeepSeek Harness 插件，视频通过 MiniMax 官方 API 用 H3 生成。</td></tr>
<tr><td width="56"><img src="https://github.com/LIUFelix2004.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/LIUFelix2004/God-minmax-H3">God MiniMax H3</a></td><td>提示词库和 Agent 技能，用分段接力做 30 秒的 H3 成片，每条提示词附 API 参数。</td></tr>
<tr><td width="56"><img src="https://github.com/10291124jcjc-dotcom.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/10291124jcjc-dotcom/ancient-rap-battle">ancient-rap-battle</a></td><td>Claude Code 技能，做历史人物说唱对决视频，可选用 MiniMax 官方 API 的 H3 对口型。</td></tr>
<tr><td width="56"><img src="https://github.com/suihe1.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/suihe1/short-drama-production">short-drama-production</a></td><td>Codex 技能，管理短剧从大纲到粗剪的全流程，H3 任务提交到 MiniMax 官方 API 或优云智算。</td></tr>
<tr><td width="56"><img src="https://github.com/gary149.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gary149/h3-game-sprites">h3-game-sprites</a></td><td>Agent 技能，把 OpenRouter 上生成的 H3 动作片段做成 2D 精灵图，每个动作一段，逐帧抠像和检查。</td></tr>
<tr><td width="56"><img src="https://github.com/guangjun5952.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/guangjun5952/minimax-h3-broll-generator">MiniMax H3 B-roll Generator</a></td><td>Codex 技能，根据逐字稿规划 B-roll、写 H3 提示词，审批后通过 MiniMax 官方 API 或秘塔生成镜头。</td></tr>
<tr><td width="56"><img src="https://github.com/gooseworks-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/gooseworks-ai/goose-skills">Goose Skills</a></td><td>面向编程 Agent 的增长营销技能库，其中的创作者组件用 fal 上的 H3 Max 生成达人口播视频。</td></tr>
<tr><td width="56"><img src="https://github.com/makevoid.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/makevoid/motion-graphics-music-video-skill">Motion Graphics Music Video Skill</a></td><td>Claude 插件，用一首音乐和一句提示词生成动画视频，可通过 fal 用 H3 Max 图生视频。</td></tr>
</table>

<a id="h3-training"></a>

### 训练

<table>
<tr><td width="56"><img src="https://github.com/IAmIronMan42.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/IAmIronMan42/MiniMax-H3-FineTuning">MiniMax-H3-FineTuning</a></td><td>基于 diffusers 实现的监督式 LoRA 训练，损失函数里包含音频。</td></tr>
<tr><td width="56"><img src="https://github.com/verl-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/verl-project/verl-omni">verl-omni</a></td><td>用 FlowGRPO 和 DiffusionNFT 对 H3 LoRA 做强化学习后训练。</td></tr>
<tr><td width="56"><img src="https://github.com/radixark.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/radixark/miles_diffusion">miles_diffusion</a></td><td>Flow-GRPO 后训练，附 H3 文生音视频方案。</td></tr>
<tr><td width="56"><img src="https://github.com/shootthesound.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/shootthesound/Fizgig">Fizgig</a></td><td>LoRA 和 LoKr 训练工作室，带 H3 预设。</td></tr>
<tr><td width="56"><img src="https://github.com/kohya-ss.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/kohya-ss/musubi-tuner">musubi-tuner</a></td><td>LoRA 训练脚本，实验性支持 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/tdrussell.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tdrussell/diffusion-pipe">diffusion-pipe</a></td><td>流水线并行的训练脚本，支持 H3 的文生图和文生音视频训练。</td></tr>
<tr><td width="56"><img src="https://github.com/bghira.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/bghira/SimpleTuner">SimpleTuner</a></td><td>图像、视频和音频模型的微调工具包，支持 H3。</td></tr>
<tr><td width="56"><img src="https://github.com/ByteDance-Seed.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ByteDance-Seed/VeOmni">VeOmni</a></td><td>字节跳动的训练框架，附 H3 FL2VA 和 Ref2VA 的配置。</td></tr>
<tr><td width="56"><img src="https://github.com/Tencent-Hunyuan.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Tencent-Hunyuan/UniRL">UniRL</a></td><td>腾讯混元的强化学习框架，H3 文生音视频目前只支持训练端。</td></tr>
<tr><td width="56"><img src="https://github.com/X-GenGroup.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/X-GenGroup/Flow-Factory">Flow-Factory</a></td><td>流匹配模型微调框架，附 H3 文生音视频、首尾帧和参考生音视频的 GRPO LoRA 配置。</td></tr>
</table>

<a id="h3-serving"></a>

### 部署与托管

<table>
<tr><td width="56"><img src="https://github.com/sgl-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/sgl-project/sglang">SGLang</a></td><td>原生 H3 服务，<a href="https://docs.sglang.ai/cookbook/diffusion/MiniMax/MiniMax-H3">cookbook</a> 里的方案从单卡 RTX 4090 到 B300、AMD MI355X 都实测过。</td></tr>
<tr><td width="56"><img src="https://github.com/vllm-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vllm-project/vllm-omni">vLLM-Omni</a></td><td>兼容 OpenAI 的 <code>/v1/videos</code> 服务，附<a href="https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md">官方 H3 方案</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/hao-ai-lab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/hao-ai-lab/uniserve">UniServe</a></td><td>FastH3 和 FastH3-Omni 的 HTTP 服务，支持 1 到 8 卡。</td></tr>
<tr><td width="56"><img src="https://github.com/EzgiTastan.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/EzgiTastan/minimax-h3-gb300-ebpf">minimax-h3-gb300-ebpf</a></td><td>SGLang 跑 H3 在 GB300 上的可复现性能测试和追踪。</td></tr>
<tr><td width="56"><img src="https://github.com/fal-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://fal.ai">H3 Max on fal</a></td><td>fal 后训练的托管模型 H3 Max、Director 和 Turbo，权重不公开，也可从 ComfyUI Partner Nodes 调用。</td></tr>
<tr><td width="56"><img src="https://github.com/jd-nuva.png?size=96" alt="" width="48"></td><td width="240"><a href="https://nuvalab.ai/minimax-h3-dedicated-deployment/">Nuva Lab</a></td><td>为单家公司提供的专属 H3 和 FastH3 部署，可运行客户自己的微调权重。</td></tr>
<tr><td width="56"><img src="https://github.com/fal-ai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://fal.ai/models/minimax/h3/text-to-video">MiniMax H3 on fal</a></td><td>fal API 上的原版 H3，支持文生视频、<a href="https://fal.ai/models/minimax/h3/image-to-video">图生视频</a>和<a href="https://fal.ai/models/minimax/h3/reference-to-video">参考生视频</a>，最高 2K。</td></tr>
<tr><td width="56"><img src="https://github.com/replicate.png?size=96" alt="" width="48"></td><td width="240"><a href="https://replicate.com/minimax/h3">MiniMax H3 on Replicate</a></td><td>Replicate 预测 API 上的 H3，支持文生视频、首尾帧和参考生视频，可选 768P 或 2K。</td></tr>
<tr><td width="56"><img src="https://github.com/WaveSpeedAI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://wavespeed.ai/models/wavespeed-ai/minimax-h3/text-to-video">MiniMax H3 on WaveSpeed</a></td><td>以 API 形式提供的 H3 开源权重，支持文生视频、图生视频和参考生视频。</td></tr>
<tr><td width="56"><img src="https://github.com/OpenRouterTeam.png?size=96" alt="" width="48"></td><td width="240"><a href="https://openrouter.ai/minimax/hailuo-3">H3 on OpenRouter</a></td><td>通过 OpenRouter 的 API 调用 H3 和 H3 Max 视频。</td></tr>
<tr><td width="56"><img src="https://github.com/cloudflare.png?size=96" alt="" width="48"></td><td width="240"><a href="https://developers.cloudflare.com/ai/models/minimax/h3/">MiniMax H3 on Cloudflare</a></td><td>Cloudflare AI 模型目录里的 H3，支持文生视频、首尾帧和参考生视频。</td></tr>
</table>

<a id="h3-guides"></a>

### 指南

<table>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="docs/run-h3-locally.md">Run H3 locally</a></td><td>本仓库的本地运行指南：显存配置、所有量化文件、组件、兼容性和实测环境。</td></tr>
<tr><td width="56"><img src="https://github.com/jd-nuva.png?size=96" alt="" width="48"></td><td width="240"><a href="https://nuvalab.ai/blog/run-minimax-h3-locally-gpu-vram-requirements">GPU and VRAM requirements</a></td><td>Nuva Lab 关于本地运行 H3 所需显卡和显存的说明，另有和 API 的<a href="https://nuvalab.ai/blog/minimax-h3-api-price-vs-self-hosting">成本对比</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/matsuo-koya.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/matsuo-koya/minimax-h3-notes">minimax-h3-notes</a></td><td>单机运行笔记。</td></tr>
<tr><td width="56"><img src="https://github.com/IvenKooLab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/IvenKooLab/minimax-h3-turing">minimax-h3-turing</a></td><td>22 GB Turing 显卡的实战手册，用 W4A8 加 Turbo。</td></tr>
<tr><td width="56"><img src="https://github.com/FlowForgeLabAi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/FlowForgeLabAi/-h3-8gb-traps">-h3-8gb-traps</a></td><td>8 GB 笔记本的实测方案，每个坑都附复现。</td></tr>
</table>

## MiniMax-H3-Image

MiniMax-H3-Image 是 H3 系列中开放权重的图像模型，一个权重同时支持文生图和多图编辑。它 6 步就能直接生成约 1600 万像素的图片，宽高比从 1:3 到 3:1，最多可用 16 张参考图。

<a id="img-official"></a>

### 官方资源

<table>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/MiniMax-H3-Image">MiniMax-H3-Image</a></td><td>官方仓库：使用指南、效果展示、本地部署和提示词编写技能。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image">MiniMaxAI/MiniMax-H3-Image</a></td><td>原始权重，支持 1600 万像素的生成和编辑，最多 16 张参考图；也可从 <a href="https://modelscope.cn/models/MiniMax/MiniMax-H3-Image">ModelScope</a> 下载。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE">MiniMaxAI/MiniMax-H3-Image-PE</a></td><td>可选的提示词扩写模型，把简短提示词扩写成模型需要的详细英文描述。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/630433fea362e7e8b51f5e2b/bsDc8lR72KhKvm-wXxYsv.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Comfy-Org/MiniMax-H3-Image">Comfy-Org/MiniMax-H3-Image</a></td><td>原生支持所用的 ComfyUI 打包权重。</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image">ComfyUI tutorial</a></td><td>ComfyUI 官方教程，附<a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json">文生图</a>和<a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_edit.json">编辑</a>模板；另见<a href="https://blog.comfy.org/p/minimax-h3-image-is-now-available">发布公告</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://platform.minimax.io/docs/guides/image-generation">MiniMax API</a></td><td>官方 API，支持文生图、图像编辑和提示词扩写；中国大陆请用 <a href="https://platform.minimax.cn/docs/guides/image-generation">platform.minimax.cn</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://hailuoai.video/">Hailuo AI</a></td><td>在网页上或<a href="https://agent.minimax.io/">桌面应用</a>里直接使用 H3-Image；中国大陆请用 <a href="https://hailuoai.com/">hailuoai.com</a> 和 <a href="https://agent.minimax.cn">agent.minimax.cn</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_image_t2i.json">ComfyUI API node</a></td><td>ComfyUI 自带的 API 节点，通过 Comfy 账号用 H3-Image 生成和编辑图片，另有<a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_image_image_edit.json">编辑模板</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/cli">MiniMax CLI</a></td><td>官方命令行工具，可用 H3-Image 生成图片，并用最多 16 张参考图编辑。</td></tr>
<tr><td width="56"><img src="https://github.com/MiniMax-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MiniMax-AI/MiniMax-MCP">MiniMax MCP</a></td><td>官方服务，让 Agent 默认用 H3-Image 生成图片，并可用最多 16 张参考图编辑。</td></tr>
</table>

<a id="img-quantizations"></a>

### 量化与格式转换

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62ecdc18b72a69615d6bd857/E4lkPz1TZNLzIFr_dR273.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/unsloth/MiniMax-H3-Image-FP8">unsloth/MiniMax-H3-Image-FP8</a></td><td>INT8 ConvRot 和 FP8 版 DiT，各 20.3 GB，Unsloth Desktop 会自动选用。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/23f3af29b8e3f4317298458cafc54454.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/LSDJesus/MiniMax-H3-Image-fp8-nvfp4">LSDJesus/MiniMax-H3-Image-fp8-nvfp4</a></td><td>给 ComfyUI 用的 FP8 scaled（21 GB）和 NVFP4（12.5 GB）；另有 Q6_K 到 Q2_K 的 <a href="https://huggingface.co/LSDJesus/MiniMax-H3-Image-GGUF">GGUF</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/672622efdf9cce918fdc713f/2wilqSZUrc1H5aGXV54Pp.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Abiray/MiniMax-H3-Image-GGUF">Abiray/MiniMax-H3-Image-GGUF</a></td><td>剪枝版 DiT 的 GGUF，6.88 GB 起，敏感层保留 FP32，可直接用 ComfyUI-GGUF 加载。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6a7d89f6560102d6b0e60442/wLhDigI2l39VFPC_OBAFK.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Hearthscale/MiniMax-H3-Image-GGUF">Hearthscale/MiniMax-H3-Image-GGUF</a></td><td>给 stable-diffusion.cpp 用的 Q8_0、Q5_K_M、Q4_K_M 和 Q3_K_M。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/69cf8fa5b7bee3980535206f/_wFUtjc_L4WrMgAdpDwKn.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/AtomicChat/MiniMax-H3-Image-GGUF">AtomicChat/MiniMax-H3-Image-GGUF</a></td><td>按逐层敏感度扫描调过的 GGUF，7.6 到 21.4 GB，附测量数据。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/c88172b8c192ca50aa900233ae6d3c13.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/lingjiu09/minimax-h3-image-gguf">lingjiu09/minimax-h3-image-gguf</a></td><td>Q3_K_M 和 Q4_0，在 8 GB 笔记本上测过。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6357da9b68659b265eb84aab/8PEgQsPKTJVQQD2_87mP9.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/tsolful/Minimax_H3_Image_INT6_INT4GPTQ">tsolful/Minimax_H3_Image_INT6_INT4GPTQ</a></td><td>给 ComfyUI 用的 GPTQ W6A8（15.6 GiB）和 W4A8（12.8 GiB）。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/4d09b9d42626184db42666bbf1642ab1.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/speedyrulz/minimax-h3-image-int8-w4a8-hybrid">speedyrulz/minimax-h3-image-int8-w4a8-hybrid</a></td><td>INT8 和 W4A8 混合，14.8 GiB，面向 16 GB 显卡。</td></tr>
<tr><td width="56"><img src="https://www.gravatar.com/avatar/d651eb3d698c95ec23696040b7c7c6c3?d=retro&size=100" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/PotatoForge/Minimax-H3-Image-W4A4">PotatoForge/Minimax-H3-Image-W4A4</a></td><td>实验性 W4A4，11.35 GB，面向较老的显卡。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1661165878439-noauth.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WaveCut/MiniMax-H3-Image-OrbitQuant-W4A8">WaveCut/MiniMax-H3-Image-OrbitQuant-W4A8</a></td><td>W4A8 的 DiT 和文本编码器，总共 27.5 GB。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1661165878439-noauth.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/WaveCut/MiniMax-H3-Image-Pruned">WaveCut/MiniMax-H3-Image-Pruned</a></td><td>去掉用不到的权重，从 135 GB 减到 95 GB，文生图结果逐位一致；附<a href="https://github.com/iamwavecut/MiniMax-H3-Image-Pruned">构建脚本</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/630cc06015433862cfc66a16/Wam_ZPEs8SLh5wtA7Xoms.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/SD-Next/MiniMax-H3-Image-SDNQ-uint4-svd-hadamard-dynamic">SD-Next SDNQ uint4</a></td><td>给 diffusers 和 SD.Next 用的 SDNQ 4 bit；OzzyGT 另有 <a href="https://huggingface.co/OzzyGT/minimax_h3_image_sdnq_dynamic_4bit">4 bit</a> 和 <a href="https://huggingface.co/OzzyGT/minimax_h3_image_sdnq_dynamic_8bit">8 bit</a> 版本。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/65273adf77bceabaab23ee5c/6h4l-LlfXHKiqR6-15P8n.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/nativ-community/MiniMax-H3-Image-4bit">nativ-community/MiniMax-H3-Image-4bit</a></td><td>给 mlx-vlm 用的 MLX 4 bit（54.5 GB），另有 <a href="https://huggingface.co/nativ-community/MiniMax-H3-Image-8bit">8 bit 版</a>。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/69bf5ffd2a6ad8b6410a5f3f/T6MxkTP8x6ZDrrQ5SMVG1.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/VisualInference/MiniMax-H3-Image-MLX-bf16">VisualInference/MiniMax-H3-Image-MLX-bf16</a></td><td>给 minimax-h3-image-mlx 用的 MLX BF16，cguldogan 另有 <a href="https://huggingface.co/cguldogan/MiniMax-H3-Image-MLX-4bit">4 bit</a> 和 <a href="https://huggingface.co/cguldogan/MiniMax-H3-Image-MLX-8bit">8 bit</a> 版本。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/635c610bfa3c172ddf40d116/pgN34290L5I0dhy0W-E8V.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Sawfwair/MiniMax-H3-Image-MLX-8bit">Sawfwair/MiniMax-H3-Image-MLX-8bit</a></td><td>给 mere.run 用的 MLX 8 bit 导出（52 GB）。</td></tr>
</table>

<a id="img-adapters"></a>

### 蒸馏与适配器

<table>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1674929746905-60f1abe7544c2adfd699860c.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step">minimax-h3-image-pdmd-2step</a></td><td>两步 pDMD 蒸馏 LoRA，附<a href="https://huggingface.co/spaces/akhaliq/minimax-h3-image-2step">在线演示</a>。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/e2e8b6608dad261a033ecf26a75cbb05.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter">Fizgig H3-Image Training Adapter</a></td><td>训练 LoRA 时一直挂着的 rank 16 适配器。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/64b90c0ce3d41dbd696b0bd0/O6FfUDzAildors6vyrvQK.jpeg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/Felldude/MiniMaxImage_hdr_diffusers_vae">Felldude/MiniMaxImage_hdr_diffusers_vae</a></td><td>解码器微调过的 VAE，以像素精度换取更多高频细节，附与原版 VAE 的对比数据。</td></tr>
</table>

<a id="img-runtimes"></a>

### 推理引擎

<table>
<tr><td width="56"><img src="https://github.com/Comfy-Org.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Comfy-Org/ComfyUI">ComfyUI</a></td><td>原生支持，附官方模板。</td></tr>
<tr><td width="56"><img src="https://github.com/sgl-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/sgl-project/sglang">SGLang</a></td><td>用 main 分支部署，<a href="https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3-Image">cookbook</a> 有从 DGX Spark 到 B300 的方案，以及给 12 到 48 GB 显卡的卸载方案。</td></tr>
<tr><td width="56"><img src="https://github.com/vllm-project.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vllm-project/vllm-omni">vLLM-Omni</a></td><td>提供服务，附官方<a href="https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image">方案</a>，包括 AMD Instinct。</td></tr>
<tr><td width="56"><img src="https://github.com/huggingface.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/huggingface/diffusers">diffusers</a></td><td>以 Modular Diffusers 模块接入，支持文生图和参考图编辑，v0.43.0 起发布，见<a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3_image">文档</a>。</td></tr>
<tr><td width="56"><img src="https://github.com/ModelTC.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ModelTC/LightX2V">LightX2V</a></td><td>支持文生图和单图或多图参考编辑，可单卡运行，也可多卡 TP/SP 并行。</td></tr>
<tr><td width="56"><img src="https://github.com/modelscope.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/modelscope/DiffSynth-Studio">DiffSynth-Studio</a></td><td>支持推理、低显存推理，以及全量或 LoRA 训练。</td></tr>
<tr><td width="56"><img src="https://github.com/leejet.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/leejet/stable-diffusion.cpp">stable-diffusion.cpp</a></td><td>纯 C/C++ 推理，首发即支持 H3-Image 和 GGUF 模型。</td></tr>
<tr><td width="56"><img src="https://github.com/unslothai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/unslothai/unsloth">Unsloth</a></td><td>在桌面应用和网页界面里运行 H3-Image，使用 INT8 或 FP8 版 DiT。</td></tr>
<tr><td width="56"><img src="https://github.com/niw.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/niw/mmh3">mmh3</a></td><td>Rust 写的推理引擎，带 CUDA 和 Metal 算子。</td></tr>
<tr><td width="56"><img src="https://github.com/Anionex.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Anionex/h3-image-mlx">h3-image-mlx</a></td><td>在 M1 到 M4 的 Mac 上用 MLX 生成和编辑，支持 W8A8，4K 出图峰值内存约 7 GB。</td></tr>
<tr><td width="56"><img src="https://github.com/felix-ab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/felix-ab/minimax-h3-image-mlx">minimax-h3-image-mlx</a></td><td>BF16 的 MLX 移植，边计算边从磁盘读下一块权重，2048×2048 只占 5 GB 内存。</td></tr>
<tr><td width="56"><img src="https://github.com/zhongkaifu.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/zhongkaifu/TensorSharp">TensorSharp</a></td><td>.NET 推理引擎，在 ggml 上跑 H3-Image 文生图，编辑还没实现。</td></tr>
<tr><td width="56"><img src="https://github.com/animede.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/animede/minimax-h3-image-server">minimax-h3-image-server</a></td><td>带网页界面的服务端，可生成和参考图编辑；默认约需 21 GB 显存，用 W4A8 加权重流式加载可降到 8 GB。</td></tr>
<tr><td width="56"><img src="https://github.com/devin-lai.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/devin-lai/MiniMax-H3-Image-Accel">MiniMax-H3-Image-Accel</a></td><td>针对 L20 和 RTX 5090 调优的 SGLang，默认 BF16，可选 FP8 降低延迟。</td></tr>
<tr><td width="56"><img src="https://github.com/Neywa-Labs.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Neywa-Labs/mlx-vlm">mlx-vlm</a></td><td>在苹果芯片上用 MLX 跑 H3-Image 文生图和图像编辑，支持 BF16、8 bit 和 4 bit 权重。</td></tr>
</table>

<a id="img-apps"></a>

### 应用

<table>
<tr><td width="56"><img src="https://github.com/mcmonkeyprojects.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/mcmonkeyprojects/SwarmUI">SwarmUI</a></td><td>网页界面，跑 H3-Image 时最多可放 16 张图做参考或编辑。</td></tr>
<tr><td width="56"><img src="https://github.com/tin2tin.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tin2tin/Pallaidium">Pallaidium</a></td><td>Blender 视频编辑器插件，支持 H3-Image 文生图和最多 10 张参考图的编辑，用 SDNQ 4 bit，24 GB 显卡放得下。</td></tr>
<tr><td width="56"><img src="https://github.com/SHoogstad.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/SHoogstad/diffusion-for-krita">diffusion-for-krita</a></td><td>以 ComfyUI 为后端的 Krita 插件，用 H3-Image 生成、局部重绘和编辑，最多 16 张图。</td></tr>
<tr><td width="56"><img src="https://github.com/stepupgaming.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/stepupgaming/styk-paint">styk-paint</a></td><td>Godot 插件，通过 ComfyUI 让 H3-Image 把 3D 简模场景逐个视角重绘，再把画面投影回场景。</td></tr>
<tr><td width="56"><img src="https://github.com/ryoppippi.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ryoppippi/mmh3-studio">mmh3-studio</a></td><td>常驻 mmh3 服务的 macOS 和浏览器客户端，出图不用每次重新加载模型。</td></tr>
<tr><td width="56"><img src="https://github.com/BesianSherifaj-AI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BesianSherifaj-AI/h3-image-editor">h3-image-editor</a></td><td>基于 ComfyUI 的 Windows 编辑工作台，用 H3-Image 生成、参考图编辑和蒙版修补，另有 Qwen、SAM、SeedVR2 等工具。</td></tr>
<tr><td width="56"><img src="https://github.com/MrJackSpade.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/MrJackSpade/ImageGenerationServer">ImageGenerationServer</a></td><td>多用户网页前端，后端接本地 ComfyUI，可用 H3-Image 生成和最多 16 张图编辑；非商用许可。</td></tr>
<tr><td width="56"><img src="https://github.com/tong-io.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/tong-io/tongflow-modal-minimax-h3-image">tongflow-modal-minimax-h3-image</a></td><td>TongFlow 插件，在 Modal 上跑 H3-Image 的生成、编辑和最多 16 张图的融合。</td></tr>
</table>

<a id="img-api"></a>

### API 集成

<table>
<tr><td width="56"><img src="https://github.com/heygen-com.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/heygen-com/hyperframes">HyperFrames</a></td><td>为 Agent 设计的 HTML 转视频工具，素材命令可通过 HeyGen 用 H3-Image 生成 2K 或 4K 图片。</td></tr>
<tr><td width="56"><img src="https://github.com/vercel.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/vercel/ai">AI SDK</a></td><td>Vercel 的 TypeScript 工具包，MiniMax provider 可用 H3-Image 及高速版生成和编辑图片。</td></tr>
<tr><td width="56"><img src="https://github.com/looplj.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/looplj/reveric">Reveric</a></td><td>小说和短剧桌面创作工作台，可通过 MiniMax 官方 API 用 H3-Image 生成和编辑图片。</td></tr>
<tr><td width="56"><img src="https://github.com/langgenius.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/langgenius/dify-official-plugins/tree/main/tools/minimax_image">Dify MiniMax Image plugin</a></td><td>Dify 工具插件，通过 MiniMax 官方 API 用 H3-Image 生成图片，并可用 1 到 16 张参考图编辑。</td></tr>
<tr><td width="56"><img src="https://github.com/ThinkFlowLab.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ThinkFlowLab/OpenDirector">OpenDirector</a></td><td>面向 Agent 的视频工作流工具，带 H3-Image 适配器，可通过 MiniMax 官方 API 用最多 16 张参考图编辑图片。</td></tr>
<tr><td width="56"><img src="https://github.com/OctaFuse.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/OctaFuse/octafuse-gateway">Octafuse Gateway</a></td><td>可自托管的 AI 网关，提供 H3-Image 生成和编辑的预设，上游是 MiniMax 兼容 OpenAI 的图片接口。</td></tr>
</table>

<a id="img-comfyui"></a>

### ComfyUI 节点与工作流

<table>
<tr><td width="56"><img src="https://github.com/Rinne414.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Rinne414/ComfyUI-MiniMaxH3-Tools">ComfyUI-MiniMaxH3-Tools</a></td><td>编码后先释放文本编码器再加载 DiT，让 H3-Image 能在 24 GB 显存加 32 GB 内存上运行。</td></tr>
<tr><td width="56"><img src="https://github.com/nkxx188.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy">ComfyUI-MiniMaxH3-Easy</a></td><td>其中的 Easy Image 节点做 H3-Image 文生图和参考图编辑，在提示词里输入 @ 插入参考图。</td></tr>
<tr><td width="56"><img src="https://github.com/pxyllvr.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/pxyllvr/ComfyUI-H3Image-MediaLoader">ComfyUI-H3Image-MediaLoader</a></td><td>加载最多 16 张参考图，每格可裁剪、旋转和调整顺序。</td></tr>
<tr><td width="56"><img src="https://github.com/DeepNeuralNerd.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/DeepNeuralNerd/H3-Video-LoRA-Loader-for-Image">H3-Video-LoRA-Loader-for-Image</a></td><td>在 H3-Image 上使用 H3 视频 LoRA。</td></tr>
<tr><td width="56"><img src="https://github.com/KadoBOT.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/KadoBOT/ComfyUI-H3-Continuity-Studio">ComfyUI-H3-Continuity-Studio</a></td><td>用 H3 视频拼接多段镜头的节点包，其中的 H3-Image 节点生成引导每个镜头的静帧。</td></tr>
<tr><td width="56"><img src="https://github.com/T8mars.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8">comfyui-minimax-h3-prompt-enhancer-T8</a></td><td>提示词扩写节点，v1.41.0 起支持 H3-Image。</td></tr>
<tr><td width="56"><img src="https://github.com/iamwavecut.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/iamwavecut/ComfyUI-OrbitQuant">ComfyUI-OrbitQuant</a></td><td>运行 OrbitQuant W4A8 版本，支持生成和最多 16 张参考图的编辑。</td></tr>
<tr><td width="56"><img src="https://github.com/Hejunlong27.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Hejunlong27/H3GIF">H3GIF</a></td><td>让 H3-Image 在一张图里画出分镜网格，再切成单帧，导出循环 GIF。</td></tr>
<tr><td width="56"><img src="https://github.com/BuffMcBigHuge.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BuffMcBigHuge/Comfiest">Comfiest</a></td><td>由 agent 驱动的 ComfyUI 工作流套件，含 H3-Image 文生图和 1、2、4 张参考图的编辑工作流。</td></tr>
<tr><td width="56"><img src="https://github.com/kuhung.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/kuhung/minimax-h3-colab-workflow">minimax-h3-colab-workflow</a></td><td>在 Colab 的 L4 上用开放权重跑 H3 和 H3-Image 的笔记本，基于 ComfyUI 量化权重和 stable-diffusion.cpp。</td></tr>
<tr><td width="56"><img src="https://github.com/Timster666.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/Timster666/comfyui-minimax-image">comfyui-minimax-image</a></td><td>启动即进入 ComfyUI、自带 H3-Image 模型的 RunPod 模板。</td></tr>
<tr><td width="56"><img src="https://github.com/yilives.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/yilives/minimax-h3-v100-guide">minimax-h3-v100-guide</a></td><td>在一张 16 GB V100 加 32 GB 内存上实测过的 H3 和 H3-Image 部署：配置、启动器和工作流。</td></tr>
</table>

<a id="img-training"></a>

### 训练

<table>
<tr><td width="56"><img src="https://github.com/shootthesound.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/shootthesound/Fizgig">Fizgig</a></td><td>LoRA 和微调工作室，v9.6 起支持 H3-Image。</td></tr>
<tr><td width="56"><img src="https://github.com/ostris.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/ostris/ai-toolkit">ai-toolkit</a></td><td>有图形界面和命令行的训练工具，数据集带控制图时可训练 H3-Image 的编辑能力。</td></tr>
</table>

<a id="img-prompting"></a>

### 提示词

<table>
<tr><td width="56"><img src="https://github.com/AI-KSK.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/AI-KSK/minimax-h3-image-prompts">minimax-h3-image-prompts</a></td><td>可复用的技能，用来写生成和编辑提示词，最多 16 张参考图。</td></tr>
<tr><td width="56"><img src="https://github.com/BeatAPI.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/BeatAPI/awesome-minimax-h3-image-prompts">awesome-minimax-h3-image-prompts</a></td><td>63 个注明出处的 H3-Image 案例，附结果图和编辑参考图；只有 5 个附了完整提示词。</td></tr>
</table>

<a id="img-benchmarks"></a>

### 评测

<table>
<tr><td width="56"><img src="https://github.com/jo-nike.png?size=96" alt="" width="48"></td><td width="240"><a href="https://github.com/jo-nike/h3-image-bench">h3-image-bench</a></td><td>在 16 GB 的 RTX 5080 上以约 1600 万像素跑两轮共 600 条打分提示词，附在线结果页。</td></tr>
<tr><td width="56"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/69c9b513833ed97d9b2f808d/xDH70MJgqXXRZn1ekMirh.png" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench">M5 Pro Mac mini benchmark</a></td><td>在 64 GB 的 M5 Pro Mac mini 上用 ComfyUI 跑 INT8 版：4096×4096 一张约 9 分钟。</td></tr>
<tr><td width="56"><img src="https://huggingface.co/avatars/9947803a52252be0e6db004495bb2a75.svg" alt="" width="48"></td><td width="240"><a href="https://huggingface.co/datasets/LIOSAM007/h3-image-12gb-tests">12 GB tests</a></td><td>12 GB 的 RTX 4070 SUPER 加 96 GB 内存跑 INT8 版：4 MP 约 11 到 12 秒，16 MP 约 41 秒。</td></tr>
</table>

## 贡献

在对应的表格里加一行：作者头像、链接到项目的名称、一句话介绍。`README.md` 里也加上同样的一行英文，然后运行 `python3 scripts/check_readme.py` 检查。

收录条件：项目自己的 README、文档或模型卡写明它是什么，并且用到了 MiniMax H3 或 MiniMax-H3-Image，本地运行或通过 API 调用都可以。测速数据、显存配置和逐个文件的说明请写进 [docs/run-h3-locally.md](docs/run-h3-locally.md)。

<a id="credits"></a>

## 致谢

这份索引离不开所有发布模型、工具、测试和文档的人。感谢 MiniMax、ComfyUI、SGLang、vLLM、NVIDIA 和 Unsloth 团队，也感谢在真实硬件上持续测试 H3 的各位独立维护者。

最初的结构和大部分资源来自社区维护的 [`wildminder/awesome-minimax-H3`](https://github.com/wildminder/awesome-minimax-H3)。本页在它的基础上，加入了 MiniMax 自己对 GitHub 和 Hugging Face 的检索结果。

特别感谢：

* [Comfy-Org](https://huggingface.co/Comfy-Org)：官方 ComfyUI 转换、工作流模板和首发支持。
* [ModelTC / LightX2V](https://github.com/ModelTC/LightX2V)：Turbo 蒸馏和公开的训练配置。
* [`Larryvrh`](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)：Turbo 推荐背后的权重对比。
* [`Kijai`](https://github.com/kijai/ComfyUI-SolAttn_triton)：NVIDIA Sol-Attn 的实现和测速记录。
* [Alibaba PAI](https://huggingface.co/alibaba-pai) 和 [`aigc-apps/VideoX-Fun`](https://github.com/aigc-apps/VideoX-Fun)：PDD 8 步加速 LoRA 和 Fun ControlNet-Union 控制分支。
* [OpenVDN](https://github.com/OpenVDN/vdn-minimax-h3)：随权重一起公开了 Video Delta Net 的训练代码。
* [FastVideo](https://huggingface.co/FastVideo)：FastH3 系列（4 步预览、V2、Trim 和 Omni）、免数据蒸馏方案和 UniServe。
* [Nuva Lab](https://nuvalab.ai/fasth3/)：FastH3 的生产验证、FastH3-Omni 的联合开发和参考数据集，以及 UniServe 测速所用的生产硬件。
* [Pika Labs](https://github.com/Pika-Labs/Pika-H3-Flex)：6 步 Flex 适配器和推理示例；[NVIDIA](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K)：公开的草稿精修模型和快速解码器，基于 LBH、LightX2V 和 LMS 的组件。
* [`IAmIronMan42`](https://github.com/IAmIronMan42/MiniMax-H3-FineTuning)：训练工作和记录下来的修复。
* [Salvatore Sanfilippo (antirez)](https://github.com/antirez)：独立的 H3 C/Metal 推理引擎。
* 上面提到的每一位量化和工作流维护者。你们的文件、测试时间和文档让本地运行 H3 方便了很多。

如果发现数字有误、链接失效或缺少兼容性说明，欢迎提 issue 或直接修正。

## 联系我们

MiniMax H3 相关问题请联系 [model@minimax.io](mailto:model@minimax.io)。
