# v0.4.2-compat.1 — prerelease notes

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
