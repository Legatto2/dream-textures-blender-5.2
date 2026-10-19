# Dream Textures — Blender 5.2 compatibility build

**Build: 0.4.2-compat.2 · 19 October 2026**

An experimental community build for **Blender 5.2.2 LTS on Windows x64**, using Blender's **Python 3.13** and an **NVIDIA RTX 5060**. Based on Dream Textures 0.4.1, commit c2622a8a9f1ae6b790cfe1d2571f814b126811b4.

Original project by **Carson Katri and the Dream Textures contributors**. Compatibility fork maintained by **Michael Heald ([Legatto2](https://github.com/Legatto2))**. See [attribution and licensing](ATTRIBUTION.md).

This revision includes the attribution notice in the installation package and improves the accompanying credits. Runtime behavior and dependency pins are unchanged from **0.4.2-compat.1**; only documentation and package/build labels change.

The ZIP contains the add-on's source, its reviewed small watermark wheel, documentation, and validation evidence. The GPU runtime and Stable Diffusion model download separately through Blender.

## Install in Blender

1. Save **Dream-Textures-0.4.2-compat2-Blender52-Windows.zip** somewhere easy to find, such as Downloads. Keep it zipped for installation.
2. Open **Blender 5.2.2 → Edit → Preferences → Add-ons**.
3. Open the menu at the upper right and choose **Install from Disk…**. Select this ZIP, then install it. If an older Dream Textures copy is enabled, disable it before replacing it.
4. Enable **Dream Textures (Blender 5.2 Compatibility)** and expand its settings.
5. Click **Install GPU Dependencies**. Allow online access in Blender if its network setting blocks installation. Progress appears in these settings; **Open Installation Log** shows the detailed output.
6. Wait for **Dependencies installed. Restart Blender to continue.** Then fully close and reopen Blender.
7. Return to the same settings. A successful runtime shows the GPU name and Torch/CUDA versions. **Check GPU Runtime** repeats its checks without downloading a model.
8. Click **Download Stable Diffusion 1.5**. The current FP16 model selection is approximately **2.74 GB**. This public model works with the token field left blank.
9. Change a Blender area to **Image Editor**, choose **View → Sidebar** (or press **N** with the pointer over that area), then open the **Dream** tab.

Setup runs through Blender's preferences. Dependencies install with Blender's embedded interpreter into Dream Textures' own runtime directory.

## First image on the RTX 5060

If ComfyUI is still running, exit its app or running console to make its GPU memory available.

Use these starting settings:

| Setting | First test |
|---|---|
| Backend | HuggingFace Diffusers |
| Model | stable-diffusion-v1-5/stable-diffusion-v1-5 |
| Image size | 512 × 512 |
| Steps | 20 |
| Batch size | 1 |
| Half Precision | Enabled |
| CPU Offload | Model — the default in this compatibility build |
| SDP Attention | Enabled |
| SDXL Refiner | Disabled |

Try the prompt **“weathered stone wall texture, even lighting, detailed surface”**, then click the generation control.

A generated image in the Image Editor completes the first Windows/GPU smoke test. These are conservative starting settings; scene complexity and other GPU applications affect available memory.

## What has been checked

**47 automated tests passed for the 0.4.2-compat.1 implementation**, retained in this revision:

- 11 tests in the official Blender 5.2.2 application: registration/lifecycle, current node interfaces, image and render color handling, brush settings, presets, and handling of a backend error.
- 14 offline backend tests using Blender's actual Python 3.13.13 and real tiny randomly initialized diffusion models: text/image generation, inpainting, depth, ControlNet, SDXL, upscaling, seamless generation, tiled decoding, previews, watermarking, and image conversion.
- 16 installer tests: isolated dependency commands, runtime validation/activation, cancellation, path/ABI checks, restart behavior, and rollback.
- 6 tests of the separate worker in Blender: interpreter/dependency isolation, NumPy results, exception transport, cancellation, and recovery after process/transport failures.

All 52 pinned packages resolve to compatible Windows Python 3.13 binary or pure-Python wheels. The CPU test runtime matches its corresponding 52 pins.

**Windows DLL loading, RTX 5060 CUDA execution, and a full pretrained-model generation still require the first test on your Windows PC.** Linux CPU tests cannot establish those results. The installer performs small CUDA matrix multiplication and convolution checks before it selects a new GPU runtime.

The detailed scope and logs are in **compatibility/BUILD_REPORT.md** and **compatibility/evidence/**.

For **0.4.2-compat.2**, verification covers the original archive and source hashes, unchanged runtime implementation and dependency pins, the new ZIP's CRCs and file manifest, and inclusion of the attribution/license notices. The original execution logs are preserved; the 47 application/runtime tests were not repeated for these documentation and build-label changes.

## Runtime changes retained from 0.4.2-compat.1

- Blender 5.2 node, brush, socket, render, lifecycle, and NumPy APIs.
- A private, pinned Python 3.13 runtime with **PyTorch 2.9.1 / CUDA 12.8**, replacing reliance on the old bundled interpreter-specific libraries.
- Background installation with cancellation, logs, validation, restart selection, and restoration of the previous validated runtime.
- Worker isolation from inherited virtual-environment paths; failures complete the waiting operation instead of leaving Blender waiting indefinitely.
- Fixes for depth prompt encoding, offload hook cleanup, tiled VAE return types, and unknown-length downloads.
- Preserved watermark support with one OpenCV provider. The bundled watermark wheel changes dependency metadata only; its code/model/license bytes remain original.
- CPU model offloading enabled by default in the Blender UI for this 8 GB target.

## If something fails

| What you see | Next step |
|---|---|
| Installation failed | Open **Installation Log** in the add-on settings and use the actual last error to diagnose it. |
| Dependencies installed but generation is unavailable | Fully restart Blender. Toggling the add-on does not switch the runtime within the same process. |
| CUDA/device check failed | Keep the log. The failed runtime is not activated. Check the NVIDIA driver/device error shown there. |
| Out of GPU memory during generation | Use 512 × 512, batch 1, Half Precision, CPU Offload **Model**, and close other GPU applications. |
| A later runtime update causes trouble | Choose **Restore Previous Runtime**, then restart Blender. |
| Two Dream Textures copies are reported | Disable the old duplicate and enable one copy of this build. |

This add-on retains its module name **dream_textures** to preserve existing preferences. Runtime data is stored under Blender's configuration folder in **dream_textures_compat**; model files use the Hugging Face cache or your linked checkpoint paths. The installer retains previous completed runtimes for rollback.

## Source and license

Dream Textures source and original contributions to this compatibility fork are **GPL-3.0-or-later**; see [LICENSE](LICENSE). The [attribution notice](ATTRIBUTION.md) credits Carson Katri and the original contributors, identifies Michael Heald as compatibility maintainer, and records the modifications. The original project README is retained as [README.upstream.md](README.upstream.md). This compatibility build's implementation and test source are included.

The bundled watermark wheel is MIT-licensed; its license and exact modification record are in **compatibility/invisible-watermark/**. See **THIRD_PARTY_NOTICES.md**.

Primary project references:

- Dream Textures 0.4.1: https://github.com/carson-katri/dream-textures/releases/tag/0.4.1
- Official PyTorch version/installation matrix: https://pytorch.org/get-started/previous-versions/
- NVIDIA CUDA GPU capabilities: https://developer.nvidia.com/cuda/gpus
- Default model: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- Blender 5.2: https://www.blender.org/releases/5-2/
