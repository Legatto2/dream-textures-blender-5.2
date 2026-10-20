# Third-party notices

## Dream Textures

This compatibility build modifies Dream Textures 0.4.1:

- Project: https://github.com/carson-katri/dream-textures
- Source commit: c2622a8a9f1ae6b790cfe1d2571f814b126811b4
- License: GNU General Public License, version 3 or any later version.
- Original project: Carson Katri and the Dream Textures contributors; original notices retained.
- Compatibility maintainer: Michael Heald (Legatto2).
- Compatibility implementation: 0.4.2-compat.1, 19 October 2026.
- Attribution and packaging revision: 0.4.2-compat.2, 19 October 2026; runtime behavior and dependency pins unchanged.
- ControlNet hotfix source revision: 0.4.2-compat.3, 20 October 2026; cached offload lifecycle and recommended-model preference corrections, with updated regression and Windows validation records.

See [ATTRIBUTION.md](ATTRIBUTION.md) for the complete original-project and compatibility-fork credits. Original contributions made for this fork are offered under GPL-3.0-or-later, consistent with the upstream add-on; [LICENSE](LICENSE) is retained unchanged.

The complete modified add-on source is included in the installation ZIP. The original 0.4.2-compat.1 implementation patch is retained as **compatibility/upstream-changes.patch**. The current source includes the subsequent 0.4.2-compat.2 attribution and packaging changes and the 0.4.2-compat.3 hotfix described in the [current validation record](compatibility/VALIDATION_2026-10-20.md).

## Bundled invisible-watermark wheel

- Original project: https://github.com/ShieldMnt/invisible-watermark
- Original distribution: invisible-watermark==0.2.0
- Original copyright: Copyright (c) 2021 ShieldMnt
- License: MIT; reproduced verbatim in **compatibility/invisible-watermark/LICENSE.txt** and retained inside the wheel.
- Local distribution: invisible-watermark==0.2.0+dreamtextures1

The dependency on opencv-python>=4.1.0.25 was changed to opencv-python-headless>=4.1.0.25 so ControlNet preprocessing and watermarking use a single OpenCV provider. The local version, distribution metadata directory, and wheel RECORD hashes reflect this metadata-only modification.

All original package source, model files, and license bytes were verified unchanged. OpenCV usage is limited to color conversion, DCT, and inverse DCT; a watermark encode/decode test passed.

Original SHA-256:
644311beed9cfe4a9a5a4a46c740f47800cef184fe2e1297f3f4542e2d992f8b

Bundled wheel SHA-256:
96c13ca5bbbd698d4fe3204d2c1bbf28b9f981ede8050cebadc9576d15fd6a0a

The official PyPI source URL, verification record, and exact METADATA diff are in **compatibility/invisible-watermark/manifest.json** and **METADATA.diff**.

## Dependencies and models downloaded during setup

Other runtime libraries are downloaded from the explicit PyPI and official PyTorch indexes. Their distributions retain their respective licenses and notices. All selected versions are listed in **requirements/compat-runtime.txt** and **requirements/compat-torch-cuda128.txt**.

Stable Diffusion model weights are downloaded separately from the selected model repository. That repository's model card and license govern the weights. No diffusion model weights are bundled in this ZIP.
