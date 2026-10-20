# Dream Textures — Blender 5.2 compatibility build

**Source revision: 0.4.2-compat.3 · 20 October 2026 · Unreleased**

An experimental community build for **Blender 5.2.2 LTS on Windows x64**, using Blender's **Python 3.13** and an **NVIDIA RTX 5060**. Based on Dream Textures 0.4.1, commit c2622a8a9f1ae6b790cfe1d2571f814b126811b4.

Original project by **Carson Katri and the Dream Textures contributors**. Compatibility fork maintained by **Michael Heald ([Legatto2](https://github.com/Legatto2))**. See [attribution and licensing](ATTRIBUTION.md).

This revision integrates the **ControlNet repeat-run hotfix**, corrects the recommended-model FP16 preference, and adapts selected improvements from **Jason Hoku's Dream Textures fork**: local model folders, projection capture/diagnostics, and backend fixes. The [fork review](compatibility/JASONHOKU_REVIEW_2026-10-20.md) records the source commits and compatibility decisions.

Michael Heald reported successful pretrained SD 1.5 generation on Windows and three consecutive ControlNet projection runs without errors after applying the original two-file hotfix to **0.4.2-compat.2**. A later screenshot check confirmed brick coverage on one selected cube face. Those Windows tests preceded the additional backports in this revision.

The build remains experimental. These results cover the tested Windows/RTX 5060 setup and basic projection; baking, broader interactive workflows, other hardware, and performance are not yet validated. See the [current validation record](compatibility/VALIDATION_2026-10-20.md) for settings, evidence scope, and remaining work.

There is currently no published GitHub release. This repository contains the next source revision; maintainers can create the named installation ZIP using the [packaging instructions](compatibility/TESTING.md#packaging). The earlier compat.2 package requires the hotfix for the reported repeat-run failure. Installing the newly built compat.3 package in Blender has not yet been checked.

The ZIP contains the add-on's source, its reviewed small watermark wheel, documentation, and validation evidence. The GPU runtime and Stable Diffusion model download separately through Blender.

## Install in Blender

1. Obtain a maintainer-built **Dream-Textures-0.4.2-compat3-Blender52-Windows.zip**, or build it using the instructions above. Keep it zipped for installation. GitHub's automatic source archive is not this prepared installation package.
2. Open **Blender 5.2.2 → Edit → Preferences → Add-ons**.
3. Open the menu at the upper right and choose **Install from Disk…**. Select this ZIP, then install it. If an older Dream Textures copy is enabled, disable it before replacing it.
4. Enable **Dream Textures (Blender 5.2 Compatibility)** and expand its settings.
5. Click **Install GPU Dependencies**. Allow online access in Blender if its network setting blocks installation. Progress appears in these settings; **Open Installation Log** shows the detailed output.
6. Wait for **Dependencies installed. Restart Blender to continue.** Then fully close and reopen Blender.
7. Return to the same settings. A successful runtime shows the GPU name and Torch/CUDA versions. **Check GPU Runtime** repeats its checks without downloading a model.
8. Click **Download Stable Diffusion 1.5**. The current FP16 model selection is approximately **2.74 GB**. This public model works with the token field left blank.
9. Change a Blender area to **Image Editor**, choose **View → Sidebar** (or press **N** with the pointer over that area), then open the **Dream** tab.

Setup runs through Blender's preferences. Dependencies install with Blender's embedded interpreter into Dream Textures' own runtime directory.

## Starting settings on the RTX 5060

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

### Windows checks reported on 20 October 2026

- GPU runtime setup followed by successful pretrained SD 1.5 image generation on Windows 11 / Blender 5.2.2 LTS / RTX 5060.
- Three consecutive Depth ControlNet projection runs without errors after applying the hotfix, with the generator left running between runs. **Seamless was Auto** for these runs.
- Basic projection onto one selected cube face, with the generated brick pattern visible on the face. A pale top strip was also present in the generated image itself.

These are maintainer-reported hardware tests and screenshot observations. The hotfix's focused automated regressions use simulated dependency/device behavior; they do not run Blender or CUDA. Reproduction commands and the distinction from historical tests are in [TESTING.md](compatibility/TESTING.md) and the [current validation record](compatibility/VALIDATION_2026-10-20.md).

### Historical automated checks

**47 automated tests passed for the 0.4.2-compat.1 implementation**. Their original sources and evidence are retained; the full suite was not repeated for this source update:

- 11 tests in the official Blender 5.2.2 application: registration/lifecycle, current node interfaces, image and render color handling, brush settings, presets, and handling of a backend error.
- 14 offline backend tests using Blender's actual Python 3.13.13 and real tiny randomly initialized diffusion models: text/image generation, inpainting, depth, ControlNet, SDXL, upscaling, seamless generation, tiled decoding, previews, watermarking, and image conversion.
- 16 installer tests: isolated dependency commands, runtime validation/activation, cancellation, path/ABI checks, restart behavior, and rollback.
- 6 tests of the separate worker in Blender: interpreter/dependency isolation, NumPy results, exception transport, cancellation, and recovery after process/transport failures.

All 52 pinned packages resolve to compatible Windows Python 3.13 binary or pure-Python wheels. The CPU test runtime matches its corresponding 52 pins.

Those Linux tests alone did not establish Windows DLL loading or CUDA generation. The subsequent maintainer-run Windows checks above now establish the basic working path on the reported setup. The installer also performs small CUDA matrix multiplication and convolution checks before it selects a new GPU runtime.

The detailed scope and logs are in **compatibility/BUILD_REPORT.md** and **compatibility/evidence/**.

For **0.4.2-compat.2**, verification covers the original archive and source hashes, unchanged runtime implementation and dependency pins, the new ZIP's CRCs and file manifest, and inclusion of the attribution/license notices. The original execution logs are preserved; the 47 application/runtime tests were not repeated for these documentation and build-label changes.

## Changes in 0.4.2-compat.3

- Remove stale offload hooks before cached pipeline components are reused and the ControlNet container is replaced. This allows the new pipeline to receive a complete set of hooks and addresses the reported later-run CPU/CUDA mismatch.
- Use `prefer_fp16_variant` in the recommended-model UI, matching the existing preference and download operator.
- Add opt-in local **Model Folders** discovery. Folder scanning runs in the background and offers recognized supported `.safetensors` checkpoints while preserving explicit manual links and their configurations.
- Save new model, ControlNet, and refiner selections by their full identifiers so a folder rescan cannot silently retarget them. Missing selected models stay unavailable until restored or explicitly replaced.
- Use an explicit depth geometry mask, handle empty/flat captures, and retain **Dream Textures Projection Depth** and **Dream Textures Projection Color** images for inspection in the Image Editor. Captures use the invoking viewport; invalid captures are rejected before UV/material changes, and render settings are restored after capture.
- Make ControlNet **Depth and Color** use the captured color image as its initial image. Resolve linked ControlNet checkpoints for image-to-image and inpainting as well as text-to-image.
- Fix the default cancellation callback, exclude incomplete ControlNet downloads from generation choices, and use the supported DPT image processor API.
- Add focused offline regressions and a Python 3.13 GitHub Actions check that builds the installation ZIP. Keep the existing dependency pins, private runtime, license notices, and historical evidence.

### Local model folders

In Dream Textures' add-on preferences, use **Model Folders → Add Model Folder** to add a checkpoint directory, then **Rescan** when files change. This section appears below **Link Checkpoint** when the runtime is ready. Discovery does not copy model weights. Recognized files appear in the model choices after the scan completes; duplicate folder paths are combined, and explicit checkpoint links retain their selected configurations. Removing a folder removes discovery from that location without deleting files.

Automatic discovery accepts supported, identifiable `.safetensors` checkpoints. Unknown or ambiguous files, including SD 2 checkpoints without enough prediction-type metadata, still need **Link Checkpoint**. Scans have directory, entry, header-size, and elapsed-time limits. A capped scan keeps the previous catalog; add a more specific directory and rescan when needed. This feature does not add Flux, Z-Image, GGUF, or a ComfyUI generation backend. See the [integration review](compatibility/JASONHOKU_REVIEW_2026-10-20.md).

Selecting a model now saves its full identifier for later rescans and file reloads. Older saved files retain their numeric interpretation over the downloaded/manually linked model list until a model is selected again; folder-discovered rows are excluded from that legacy interpretation. A discovered file can later be explicitly linked to choose its configuration while retaining its saved identity.

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
| A second or later ControlNet run reports CUDA inputs with CPU weights | Use the compat.3 source fix or the verified compat.2 hotfix, then fully restart Blender. If it recurs, retain the actual traceback and settings. |
| A projected face has a gray cap, background, or an unexpected object | Inspect the generated image itself. The same pixels in that image indicate generated content to refine; they do not establish a new material or UV failure. |
| Projection reports an empty depth capture | Inspect **Dream Textures Projection Depth** in the Image Editor and check the invoking viewport's visible geometry. Empty captures stop before UV/material changes. |
| Projection reports uniform depth | A front-on flat face can produce uniform depth. Generation can proceed; a slight viewing angle provides more shape information when needed. |
| A model folder omits a file | Check scan status and the checkpoint format. Automatic discovery is conservative; use **Link Checkpoint** with the appropriate configuration for an ambiguous supported checkpoint. |
| Successive images look different | Check **Random Seed**. Disable it and choose a fixed seed when comparing prompt or framing changes. **Seamless Auto** is a separate setting. |
| A later runtime update causes trouble | Choose **Restore Previous Runtime**, then restart Blender. |
| Two Dream Textures copies are reported | Disable the old duplicate and enable one copy of this build. |

This add-on retains its module name **dream_textures** to preserve existing preferences. Runtime data is stored under Blender's configuration folder in **dream_textures_compat**; model files use the Hugging Face cache or your linked checkpoint paths. The installer retains previous completed runtimes for rollback.

## Source and license

Dream Textures source and original contributions to this compatibility fork are **GPL-3.0-or-later**; see [LICENSE](LICENSE). The [attribution notice](ATTRIBUTION.md) credits Carson Katri and the original contributors, Jason Hoku for the adapted changes, and Michael Heald as compatibility maintainer. The original project README is retained as [README.upstream.md](README.upstream.md). This compatibility build's implementation and test source are included.

The bundled watermark wheel is MIT-licensed; its license and exact modification record are in **compatibility/invisible-watermark/**. See **THIRD_PARTY_NOTICES.md**.

Primary project references:

- Dream Textures 0.4.1: https://github.com/carson-katri/dream-textures/releases/tag/0.4.1
- Official PyTorch version/installation matrix: https://pytorch.org/get-started/previous-versions/
- NVIDIA CUDA GPU capabilities: https://developer.nvidia.com/cuda/gpus
- Default model: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- Blender 5.2: https://www.blender.org/releases/5-2/
