# v0.4.2-compat.2 — prerelease notes

**Experimental community build · 19 October 2026**

An attribution and packaging revision of **v0.4.2-compat.1** for **Blender 5.2.2 LTS / Windows x64 / Python 3.13 / NVIDIA RTX 5060**. Blender continues to display the add-on version as **0.4.2**.

## Installation asset

- Filename: **Dream-Textures-0.4.2-compat2-Blender52-Windows.zip**
- Size: **40,073,349 bytes**
- SHA-256: `2a61b1c45292c298a0a7d3bc8106e631b39e9eb1e64ad8079fc18d6691225ba8`

The asset contains the add-on source, reviewed watermark wheel, tests, documentation, build evidence, and the complete [attribution notice](../ATTRIBUTION.md). GPU dependencies and diffusion model weights download separately. Follow the [installation guide](../README.md).

## Changes in this revision

- Include a root attribution notice in the installation ZIP, crediting **Carson Katri and the Dream Textures contributors**, with **Michael Heald (Legatto2)** identified as compatibility maintainer.
- Link the canonical notice from the installation guide, third-party notices, and GitHub attribution page.
- Update documentation and the package/build labels to **0.4.2-compat.2**.
- Preserve the original GPL-3.0-or-later license text, upstream README, MIT watermark notice/wheel, dependency pins, and historical validation evidence.

Runtime behavior is unchanged. `version.py` changes only the compatibility-build string; the packaging helper changes only its manifest build label.

## Verification and remaining acceptance

The original **40,070,223-byte** archive matched SHA-256 `2cb7f5701c420e1f9b3c8515380d419c489b7db985a148bb510d9c02a4f58bed`; its 179 ZIP entries and 178 manifest records passed validation. The complete local baseline matched all 185 source files at commit `773501c6867f455e120ee9a134c216da6baeedee`.

Revision checks cover source and dependency comparisons, the two build-label-only Python changes, the complete changed-file set, root notices and relative links, the regenerated file manifest, ZIP CRCs, and the single `dream_textures/` archive root.

The **47 automated application/runtime test results belong to v0.4.2-compat.1** and are preserved unchanged. They were not rerun for this attribution and packaging revision. **Actual Windows DLL loading, RTX 5060 CUDA operations, pretrained-model generation, performance, and interactive painting/projection/viewport acceptance remain pending.**

See the [build report](../compatibility/BUILD_REPORT.md), [historical test evidence](../compatibility/evidence/), and [third-party notices](../THIRD_PARTY_NOTICES.md).

---

## Previous package record

The following v0.4.2-compat.1 notes are retained as the record of the original package and its validation.

### v0.4.2-compat.1 — prerelease notes

**Experimental community build · 19 October 2026**

Targets **Blender 5.2.2 LTS / Windows x64 / Python 3.13 / NVIDIA RTX 5060**, based on Dream Textures 0.4.1. Blender displays the add-on version as **0.4.2**.

## Installation asset

- Filename: **Dream-Textures-0.4.2-compat1-Blender52-Windows.zip**
- Size: **40,070,223 bytes**
- SHA-256: `2cb7f5701c420e1f9b3c8515380d419c489b7db985a148bb510d9c02a4f58bed`

The asset includes the add-on source, reviewed watermark wheel, tests, documentation, and build evidence. GPU dependencies and diffusion model weights download separately. Follow the [installation guide](../README.md).

## Changes

- Updates Blender node, brush, render, lifecycle, and NumPy compatibility.
- Adds a private Python 3.13 runtime with pinned PyTorch 2.9.1 / CUDA 12.8 dependencies, validation before activation, cancellation, logs, and rollback.
- Improves worker isolation and recovery, depth prompt encoding, offload cleanup, tiled VAE handling, and download progress.
- Defaults CPU Offload to **Model** for the intended 8 GB GPU target.

## Verification and remaining acceptance

**47 automated tests passed:** 11 Blender frontend tests, 14 backend tests, 16 installer tests, and 6 Blender worker integration tests. Execution used **official Blender 5.2.2 Linux, Python 3.13.13, and CPU Torch**. Backend inference used tiny random-weight models offline.

All **52 production dependency pins** resolve to Windows Python 3.13 wheels. The corresponding CPU runtime matches its 52 pins. Blender's actual ZIP installer passed installation, enable, and cleanup checks. Documentation and evidence were subsequently added; all production files match the tested copy.

**Pending:** actual Windows DLL loading, RTX 5060 CUDA operations, pretrained-model generation, performance, and full interactive painting/projection/viewport validation. Wheel resolution and Linux CPU tests do not establish these results. The Windows installer performs small CUDA checks before activating a runtime.

See the [build report](../compatibility/BUILD_REPORT.md), [evidence](../compatibility/evidence/), and [test instructions](../compatibility/TESTING.md).

Original Dream Textures attribution and **GPL-3.0-or-later** licensing are retained. The bundled watermark wheel retains its MIT license; see [third-party notices](../THIRD_PARTY_NOTICES.md).

Original project by **Carson Katri and the Dream Textures contributors**. Compatibility fork maintained by **Michael Heald (Legatto2)**; see [attribution and licensing](ATTRIBUTION.md).

