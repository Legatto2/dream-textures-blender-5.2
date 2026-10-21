# Blender 5.2 setup and troubleshooting

This guide covers the **0.4.2-compat.3** community build for **Blender 5.2.2 LTS on Windows x64**, using Blender's **Python 3.13** and targeting the **NVIDIA RTX 5060**.

For the feature overview, see the [main README](../README.md). The other workflow guides in this directory originate from Dream Textures; use the installation and model setup instructions here for this fork.

## Installation

There is currently no published GitHub release. Obtain a maintainer-built **Dream-Textures-0.4.2-compat3-Blender52-Windows.zip**, or create it using the [packaging instructions](../compatibility/TESTING.md#packaging). GitHub's automatic source archive is not the prepared installation package.

1. Keep the prepared installation ZIP zipped.
2. Open **Blender 5.2.2 → Edit → Preferences → Add-ons**.
3. Open the menu at the upper right and choose **Install from Disk…**. Select the ZIP and install it.
4. Enable **Dream Textures (Blender 5.2 Compatibility)** and expand its settings.
5. Click **Install GPU Dependencies**. Allow online access in Blender if its network setting blocks installation. Progress appears in the settings; **Open Installation Log** shows detailed output.
6. Wait for **Dependencies installed. Restart Blender to continue.** Fully close and reopen Blender.
7. Return to the add-on settings. The runtime status shows the GPU name and Torch/CUDA versions. **Check GPU Runtime** repeats the runtime checks without downloading a model.
8. Click **Download Stable Diffusion 1.5**. The default model is `stable-diffusion-v1-5/stable-diffusion-v1-5`; this public model works with the token field left blank.
9. Change a Blender area to **Image Editor**, choose **View → Sidebar** (or press **N** with the pointer over that area), then open the **Dream** tab.

The add-on, GPU dependencies, and model weights are separate downloads. Setup uses Blender's embedded interpreter and installs dependencies into Dream Textures' own runtime directory.

### Updating an existing installation

Disable an older Dream Textures copy before replacing it with the prepared ZIP, then fully restart Blender. Keep only one copy enabled. An existing compatible runtime can be reused; check its status before starting another dependency installation.

The add-on keeps the module name `dream_textures` so existing preferences remain compatible. Runtime data is stored under Blender's configuration folder in `dream_textures_compat`; models use the Hugging Face cache or your linked checkpoint paths. Previous completed runtimes are retained for **Restore Previous Runtime**.

## Starting settings for the RTX 5060

Close other applications that are using GPU memory, including a running ComfyUI process, before starting a demanding generation.

| Setting | Starting value |
|---|---|
| Backend | HuggingFace Diffusers |
| Model | stable-diffusion-v1-5/stable-diffusion-v1-5 |
| Image size | 512 × 512 |
| Steps | 20 |
| Batch size | 1 |
| Half Precision | Enabled |
| CPU Offload | Model — the default in this build |
| SDP Attention | Enabled |
| SDXL Refiner | Disabled |

Try **“weathered stone wall texture, even lighting, detailed surface”**, then click **Generate**. Scene complexity, image size, and other GPU applications affect available memory.

**Random Seed** selects a different seed for each generation. Disable it and choose a fixed seed when comparing prompts or framing. **Seamless** controls tiling separately.

## Local model folders

In the add-on preferences, use **Model Folders → Add Model Folder** to add a checkpoint directory, then **Rescan** when files change. This section appears below **Link Checkpoint** when the runtime is ready.

Recognized files appear in the model choices after the background scan completes. Discovery does not copy model weights. Removing a folder stops discovery there without deleting files.

Automatic discovery accepts supported, identifiable `.safetensors` checkpoints. Unknown or ambiguous files, including SD 2 checkpoints without enough prediction-type metadata, need **Link Checkpoint** with the appropriate configuration.

Scans limit filesystem work and checkpoint-header reading. If a limit is reached, check the status message, choose smaller folders, and rescan. Previous results for registered folders are retained where valid, while downloaded and manually linked models can still refresh.

New model selections are saved by their full identifiers and preserved across rescans. Older saved files use the legacy numeric selection until a model is selected again. If a selected model becomes unavailable, select an available model rather than relying on its old position in the list. Explicit checkpoint links retain their chosen configurations.

Folder discovery works with compatible checkpoint files; it does not add Flux, Z-Image, GGUF, or a ComfyUI generation backend. Implementation details are in the [fork integration review](../compatibility/JASONHOKU_REVIEW_2026-10-20.md).

## Projection inputs

Projection captures the invoking 3D Viewport. Use **Depth** to guide generation from geometry, or **Depth and Color** to include the captured viewport color as the initial image.

The captured inputs remain available in the Image Editor as **Dream Textures Projection Depth** and **Dream Textures Projection Color**. Inspect these alongside the generated image when adjusting framing or diagnosing an unexpected result.

See the [texture projection guide](TEXTURE_PROJECTION.md) for selecting faces and using **Project Dream Texture**.

## Troubleshooting

| What you see | Next step |
|---|---|
| Installation failed | Open **Installation Log** in the add-on settings and inspect the last error. |
| Dependencies installed but generation is unavailable | Fully restart Blender. Toggling the add-on does not switch the runtime within the same process. |
| CUDA/device check failed | Keep the log and check the NVIDIA driver/device error shown there. A failed runtime is not activated. |
| Out of GPU memory | Use 512 × 512, batch 1, Half Precision, and CPU Offload **Model**. Close other GPU applications. |
| A later ControlNet run reports CUDA inputs with CPU weights | Install the current compatibility build and fully restart Blender. If it recurs, include the traceback and generation settings in a bug report. |
| A projected face contains an unexpected cap, background, or object | Inspect the generated image itself. If it contains the same pixels, adjust the prompt or input framing. |
| Projection reports an empty depth capture | Inspect **Dream Textures Projection Depth** and check that geometry is visible in the invoking viewport. Empty captures stop before UV/material changes. |
| Projection reports uniform depth | A front-on flat face can produce uniform depth. Generation can proceed; a slight viewing angle provides more shape information when needed. |
| A model folder omits a file | Check scan status and the checkpoint format. Use **Link Checkpoint** with the appropriate configuration for an ambiguous supported checkpoint. |
| Successive images look different | Disable **Random Seed** and choose a fixed seed when comparing settings. Seamless is a separate option. |
| A runtime update causes trouble | Choose **Restore Previous Runtime**, then restart Blender. |
| Two Dream Textures copies are reported | Disable the old duplicate and enable one copy of this build. |

On Windows, **Window → Toggle System Console** exposes additional Blender error output. When [reporting a problem](https://github.com/Legatto2/dream-textures-blender-5.2/issues), include the Blender and add-on versions, operating system, GPU, reproduction steps, and relevant log excerpt. Remove credentials from shared logs.
