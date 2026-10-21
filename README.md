![Dream Textures — Stable Diffusion built into Blender](docs/assets/banner.png)

# Dream Textures for Blender 5.2

Create textures, concept art, background assets, and more directly in Blender with Stable Diffusion. This community fork brings [Dream Textures](https://github.com/carson-katri/dream-textures) to **Blender 5.2.2 LTS on Windows**, with an updated GPU setup and improvements to model management and texture projection.

**Current source: 0.4.2-compat.3 · Experimental community build**

- Generate images from text prompts or an existing image.
- Create tileable textures with the **Seamless** options.
- Project generated textures onto models and scenes using viewport depth.
- Inpaint, outpaint, upscale, and reuse prompts from generation history.
- Build effects with the Dream Textures node system and render workflows.
- Run local models on your own GPU.

## What this fork adds

- **Blender 5.2 compatibility.** Updates to node, brush, render, and add-on lifecycle APIs.
- **GPU setup inside Blender.** A private Python 3.13 runtime with PyTorch 2.9.1 / CUDA 12.8, installed through the add-on preferences. Includes progress, logs, cancellation, and runtime rollback.
- **Defaults for the RTX 5060.** CPU model offloading is enabled by default to help manage GPU memory.
- **Local model folders.** Discover supported `.safetensors` checkpoints in existing folders without copying the weights, and keep model selections across rescans.
- **Projection improvements.** Better depth capture, corrected **Depth and Color** input, and depth/color images that can be inspected in the Image Editor.
- **Generation and UI fixes.** Improvements to repeated ControlNet generation, linked checkpoints, model downloads, cancellation, and FP16 model preferences.

Selected improvements are adapted from [Jason Hoku's Dream Textures fork](https://github.com/JasonHoku/dream-textures).

## Installation

This build targets **Windows x64, Blender 5.2.2 LTS / Python 3.13, and the NVIDIA RTX 5060**. The GPU runtime and model weights download separately from the add-on.

1. Obtain or build the prepared installation ZIP using the [Blender 5.2 setup guide](docs/BLENDER_5_2.md#installation).
2. In Blender, open **Edit → Preferences → Add-ons → Install from Disk…** and select the ZIP.
3. Enable **Dream Textures (Blender 5.2 Compatibility)**, expand its settings, and click **Install GPU Dependencies**. Restart Blender when installation finishes.
4. Click **Download Stable Diffusion 1.5**, then open **Image Editor → Sidebar → Dream** to generate an image.

The [setup guide](docs/BLENDER_5_2.md) also covers recommended starting settings, local model folders, updating the add-on, and troubleshooting.

## Usage

These guides introduce Dream Textures' core workflows. Use the [Blender 5.2 setup guide](docs/BLENDER_5_2.md) for this fork's installation and model setup.

### [Image generation](docs/IMAGE_GENERATION.md)

Create textures, concept art, and other images from a text prompt. Use an existing image to guide the composition, or enable seamless generation for tileable textures.

![Image generation workflow](docs/assets/image_generation.png)

### [Texture projection](docs/TEXTURE_PROJECTION.md)

Generate a texture from viewport depth and project it onto selected faces. **Depth and Color** also uses the current viewport's color as a starting image.

![Texture projection workflow](docs/assets/texture_projection.png)

### [Inpainting and outpainting](docs/INPAINT_OUTPAINT.md)

Replace parts of an image, refine an existing texture, or extend an image beyond its original edges.

![Inpainting and outpainting workflow](docs/assets/inpaint_outpaint.png)

### [Render workflows](docs/RENDER_PASS.md)

Use Dream Textures nodes and render integration to create effects and restyle scenes or animations.

![Scene and generated render results](docs/assets/render_pass.png)

### [AI upscaling](docs/AI_UPSCALING.md)

Increase image resolution by 4× with the Stable Diffusion upscaler.

![AI upscaling workflow](docs/assets/upscale.png)

### [History](docs/HISTORY.md)

Recall previous generation settings, and import or export prompts for later use.

## Help and contributing

Report problems or suggest improvements in [this fork's issue tracker](https://github.com/Legatto2/dream-textures-blender-5.2/issues). For setup problems, start with [troubleshooting](docs/BLENDER_5_2.md#troubleshooting). The [Dream Textures Discord](https://discord.gg/EmDJ8CaWZ7) is also available for community discussion.

See the [contribution guide](https://github.com/Legatto2/dream-textures-blender-5.2/blob/blender-5.2-compat/.github/CONTRIBUTING.md), [revision notes](https://github.com/Legatto2/dream-textures-blender-5.2/blob/blender-5.2-compat/.github/RELEASE_NOTES.md), and [compatibility development notes](compatibility/README.md) for project details.

## Credits and license

Dream Textures was created by **Carson Katri and the Dream Textures contributors**. This compatibility fork is maintained by **Michael Heald ([Legatto2](https://github.com/Legatto2))**, with adapted improvements from **Jason Hoku**.

Source is licensed under **GPL-3.0-or-later**. See [LICENSE](LICENSE), [attribution](ATTRIBUTION.md), and [third-party notices](THIRD_PARTY_NOTICES.md). The [original Dream Textures README](README.upstream.md) is preserved.
