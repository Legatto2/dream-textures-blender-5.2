# 0.4.2-compat.3 — unreleased source revision

**Experimental community build · 20 October 2026**

Integrates the original two-file hotfix tested on **compat.2 / Windows 11 / Blender 5.2.2 LTS / RTX 5060** and selected improvements from **Jason Hoku's Dream Textures fork**. Blender continues to display the add-on version as **0.4.2**. The compatibility-build identifier is **0.4.2-compat.3**; dependency pins are unchanged. The new backports were added after the reported Windows test session.

## Changes

- Remove stale pipeline and component offload hooks before cached models are reused or their ControlNet wrapper is replaced. Fresh hooks can then cover the complete reconstructed pipeline, addressing the reported CPU/CUDA mismatch on later ControlNet runs.
- Replace the stale `prefer_fp16_revision` UI reference with the existing `prefer_fp16_variant` preference and operator field.
- Add **Model Folders** preferences for opt-in background discovery of confidently supported `.safetensors` checkpoints. Bound filesystem/header work, skip malformed/ambiguous files, cache unchanged classifications, and preserve explicit checkpoint configurations.
- Store newly chosen model/ControlNet/refiner selections by full identifier. Rescans preserve the choice, and missing models remain unavailable. Existing numeric selections retain their legacy downloaded/linked-model interpretation until selected again.
- Capture depth from the invoking viewport with an explicit geometry mask, normalize finite covered pixels, and handle empty or uniform depth. Keep inspectable add-on-owned depth/color images, restore render/GPU state on errors, and reject invalid captures before UV/material changes.
- Make ControlNet **Depth and Color** explicitly use an image-to-image task with consistently oriented color input. Resolve linked ControlNet files for image-to-image and inpainting, matching text-to-image behavior.
- Fix the default cancellation callback, require supported non-empty ControlNet weight files for generation selection, provide an incomplete-download error, and select the supported DPT image processor API.
- Add focused offline regressions and a Python 3.13 Actions workflow for source compilation, tests, and package/manifest validation. Official actions are pinned to commits; GPU dependencies stay in the private installer.
- Update installation guidance, attribution, build metadata, and validation records to reflect the maintainer's Windows tests.

See the [JasonHoku review](../compatibility/JASONHOKU_REVIEW_2026-10-20.md) for exact source commits and decisions. The existing cache/offload repair, private runtime, and Torch 2.9.1/CUDA 12.8 stack are retained. Flux, Z-Image, GGUF/quantization, ComfyUI backend work, and camera/aspect-fit projection are separate migrations; no target-machine acceptance for them is claimed.

## Validation scope

The maintainer reported pretrained SD 1.5 generation on the target Windows machine, followed by **three consecutive ControlNet projections without errors after the hotfix**. The generator remained running between those projections; **Seamless was Auto**. Later screenshots confirmed generated bricks on a selected cube face, with a pale cap traced to the source image itself.

The combined local check passed **83 tests across seven suites** on Linux Python 3.12.14 / NumPy 2.3.5. All **108 Python files** compiled. These focused suites use temporary files, NumPy, and simulated dependencies/devices/Blender boundaries. They are separate from both the maintainer's actual Windows tests and the historical 47-test compat.1 suite. The full Blender/real-inference suite was not rerun for this revision. See the [current validation record](../compatibility/VALIDATION_2026-10-20.md), [combined offline output](../compatibility/evidence/offline-backports-2026-10-20.log), and [reproduction instructions](../compatibility/TESTING.md).

**Remaining acceptance:** installing the newly assembled compat.3 ZIP in Blender, the new model-folder UI and capture/Depth and Color changes, baking/reprojection onto existing UV layouts, broader interactive painting/projection/viewport use, quantitative performance, and other hardware. The working one-face projection does not establish seam-free or production-ready texture quality.

## Package status

This is an **unreleased source revision**. No GitHub release has been published. The packaging helper produces **Dream-Textures-0.4.2-compat3-Blender52-Windows.zip** when given that output name. Its manifest is regenerated from the current source; the existing compat.2 archive remains the historical package and does not contain this fix.

### Local package integrity check

The source was packaged and verified locally on 20 October 2026. This is a build-validation artifact, not a published release or a completed Blender installation test.

- Filename: **Dream-Textures-0.4.2-compat3-Blender52-Windows.zip**
- Size: **40,138,917 bytes**
- SHA-256: `12b760fa228204dbb517cfd100bb00da75c966d3ecc89a012995c24873eaebdd`
- **197 ZIP entries** and **196 manifest file records** verified against the working source; all CRCs passed, with one `dream_textures/` root and the attribution/license notices included.


---

## Historical package records

The following notes preserve the compat.2 and compat.1 package records. Their pending-acceptance statements describe **19 October 2026**; the current results are recorded above.

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

