# Dream Textures for Blender 5.2

A community compatibility build of [Dream Textures](https://github.com/carson-katri/dream-textures), targeting **Blender 5.2.2 LTS, Windows x64, Python 3.13, and NVIDIA RTX 5060**.

Original project: **Carson Katri and the Dream Textures contributors**. Compatibility fork maintained by **Michael Heald ([Legatto2](https://github.com/Legatto2))**. See [attribution and licensing](ATTRIBUTION.md).

**Status: experimental.** Build **0.4.2-compat.2** is an attribution and packaging revision of **0.4.2-compat.1**, whose implementation passed 47 automated tests using the official Blender 5.2.2 Linux application and its embedded Python. Windows wheel resolution also passed. Runtime behavior and dependency pins are unchanged; the revision was verified through source hashes, build metadata, ZIP integrity, and license/attribution inclusion. **Windows DLL loading, RTX 5060 CUDA execution, and generation with a pretrained model still require target-machine validation.**

## Install

The planned prerelease is **v0.4.2-compat.2**. When available on the [Releases page](https://github.com/Legatto2/dream-textures-blender-5.2/releases), download **Dream-Textures-0.4.2-compat2-Blender52-Windows.zip**. This named installation asset contains the required `dream_textures/` folder and the complete [attribution notice](../ATTRIBUTION.md).

1. In Blender, open **Edit → Preferences → Add-ons → Install from Disk…** and select the ZIP.
2. Enable **Dream Textures (Blender 5.2 Compatibility)** and expand its settings.
3. Click **Install GPU Dependencies**, allow the installation to finish, then fully restart Blender.
4. Run **Check GPU Runtime**. Download or link a model, then open **Image Editor → Sidebar → Dream**.

For an RTX 5060 first test, use **512 × 512, batch 1, Half Precision, and CPU Offload: Model**. The GPU runtime and model download separately.

See the [complete installation guide](../README.md), [prerelease notes](RELEASE_NOTES.md), and [build report with validation evidence](../compatibility/BUILD_REPORT.md).

## Development and provenance

The port updates Blender APIs and installs a private, pinned runtime using **PyTorch 2.9.1 / CUDA 12.8**. The worker keeps AI dependencies separate from Blender's frontend Python environment. See [Contributing](CONTRIBUTING.md) and [test instructions](../compatibility/TESTING.md).

This work is based on **Dream Textures 0.4.1** by **Carson Katri and contributors**, upstream commit `c2622a8a9f1ae6b790cfe1d2571f814b126811b4`. It is a community build, with source under **GPL-3.0-or-later**. See [LICENSE](../LICENSE) and [third-party notices](../THIRD_PARTY_NOTICES.md).

The [original README](../README.upstream.md) and `docs/` preserve upstream documentation. Their older installation and dependency directions are historical; use this build's [installation guide](../README.md).
