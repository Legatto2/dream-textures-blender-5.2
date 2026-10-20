# Dream Textures for Blender 5.2

A community compatibility build of [Dream Textures](https://github.com/carson-katri/dream-textures), targeting **Blender 5.2.2 LTS, Windows x64, Python 3.13, and NVIDIA RTX 5060**.

Original project: **Carson Katri and the Dream Textures contributors**. Compatibility fork maintained by **Michael Heald ([Legatto2](https://github.com/Legatto2))**. See [attribution and licensing](../ATTRIBUTION.md).

**Current source: 0.4.2-compat.3, unreleased, 20 October 2026. Status: experimental.** This revision integrates the cached ControlNet offload hotfix and adapts selected improvements from **Jason Hoku's fork**: background model-folder discovery, projection input fixes and diagnostic images, linked ControlNet dispatch, and download/cancellation fixes. The FP16 recommendation UI is also corrected. Runtime pins remain unchanged; see the [fork review and integration decisions](../compatibility/JASONHOKU_REVIEW_2026-10-20.md).

## Current validation

Michael Heald reported successful pretrained SD 1.5 image generation on Windows 11 / Blender 5.2.2 LTS / RTX 5060. After applying the same two-file hotfix to compat.2, he completed **three consecutive ControlNet projection runs without errors**, leaving the generator running between runs. **Seamless was Auto** for those runs. A later screenshot check confirmed basic projection of a generated brick pattern onto one selected cube face.

The pale strip at the top of the projected face was also visible in the generated image: the model created a gray cap above the bricks. The basic projection check passed; clean edge-to-edge brick content remains a texture-quality task. Baking, broader interactive workflows, performance, and other systems have not been validated by these checks.

The Windows results used **compat.2 plus the original two-file hotfix**, before the additional JasonHoku backports. The historical **47 automated tests** belong to the compat.1 implementation and have not been rerun in full for this update. The current focused suites exercise source against temporary model files and simulated dependency/device/Blender behavior; they do not execute Blender or CUDA. See the [current validation record](../compatibility/VALIDATION_2026-10-20.md), [build report](../compatibility/BUILD_REPORT.md), and [test instructions](../compatibility/TESTING.md).

## Install

There is currently **no published GitHub release**. The repository contains the next source revision. Maintainers can create **Dream-Textures-0.4.2-compat3-Blender52-Windows.zip** using the [packaging instructions](../compatibility/TESTING.md#packaging). Installation of that newly built package in Blender remains to be checked. GitHub's automatic source ZIP is not the prepared add-on package.

1. In Blender, open **Edit → Preferences → Add-ons → Install from Disk…** and select the prepared ZIP.
2. Enable **Dream Textures (Blender 5.2 Compatibility)** and expand its settings.
3. On first setup, click **Install GPU Dependencies**, allow the installation to finish, then fully restart Blender. Existing validated dependencies do not need to be downloaded again for this source hotfix.
4. Run **Check GPU Runtime**. Download or link a model, then open **Image Editor → Sidebar → Dream**.

For an RTX 5060 test, use **512 × 512, batch 1, Half Precision, and CPU Offload: Model**. The GPU runtime and model download separately.

See the [complete installation guide](../README.md) and [revision notes](RELEASE_NOTES.md).

## Development and provenance

The port updates Blender APIs and installs a private, pinned runtime using **PyTorch 2.9.1 / CUDA 12.8**. The worker keeps AI dependencies separate from Blender's frontend Python environment. See [Contributing](CONTRIBUTING.md).

This work is based on **Dream Textures 0.4.1** by **Carson Katri and contributors**, upstream commit `c2622a8a9f1ae6b790cfe1d2571f814b126811b4`. Additional changes adapt **Jason Hoku's** commits `32c7870`, `63c19fd`, and `84fdcfc`, reviewed through snapshot `50d53f0`. It is a community build, with source under **GPL-3.0-or-later**. See [LICENSE](../LICENSE) and [third-party notices](../THIRD_PARTY_NOTICES.md).

The [original README](../README.upstream.md) and `docs/` preserve upstream documentation. Their older installation and dependency directions are historical; use this build's [installation guide](../README.md).
