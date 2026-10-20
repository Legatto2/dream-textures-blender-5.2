# Attribution and licensing

## Original Dream Textures project

Dream Textures was created by **Carson Katri**, with contributions from the [Dream Textures contributors](https://github.com/carson-katri/dream-textures/graphs/contributors).

This compatibility fork is based on [Dream Textures 0.4.1](https://github.com/carson-katri/dream-textures/releases/tag/0.4.1), source commit `c2622a8a9f1ae6b790cfe1d2571f814b126811b4`. Original source notices, the [GPL license text](LICENSE), and the [upstream README](README.upstream.md) are retained. Credit for the original add-on remains with its authors and contributors.

## Blender 5.2 compatibility fork

**Maintainer: Michael Heald ([Legatto2](https://github.com/Legatto2)).**

The compatibility work dated **19 October 2026** updates Blender APIs, the isolated Python/GPU dependency installer, worker reliability, packaging, documentation, and tests. Its scope and validation are recorded in the [build report](compatibility/BUILD_REPORT.md).

Build **0.4.2-compat.2**, also dated **19 October 2026**, is an attribution and packaging revision of **0.4.2-compat.1**. It includes these credits with the installation ZIP, adds direct credits to the installation guide, and updates the package/build labels. Runtime behavior and dependency pins are unchanged.

The **0.4.2-compat.3** source revision, dated **20 October 2026**, adds the cached ControlNet offload lifecycle fix, corrects the FP16 preference used by the recommended-model UI, and records the Windows GPU and projection checks reported by Michael Heald. It also adapts the Jason Hoku contributions credited below. Its regression tests and validation scope are described in the [current validation record](compatibility/VALIDATION_2026-10-20.md). The upstream license and third-party notices remain applicable.

This is an independently maintained community fork. The original project authors have not endorsed or validated this compatibility release.

Original contributions made for this compatibility fork are offered under the **GNU General Public License, version 3 or any later version (GPL-3.0-or-later)**, consistent with the upstream add-on. Existing copyright and license notices remain applicable. This notice identifies the compatibility maintainer and modifications; it does not claim ownership of the original project or replace its license.

## Adapted contributions from Jason Hoku

**Jason Hoku ([JasonHoku](https://github.com/JasonHoku))** is credited for improvements in [JasonHoku/dream-textures](https://github.com/JasonHoku/dream-textures), reviewed at commit [`50d53f0724f94f1c8449ed749a711de4cd2991e4`](https://github.com/JasonHoku/dream-textures/commit/50d53f0724f94f1c8449ed749a711de4cd2991e4). This revision adapts work from:

- [`32c7870`](https://github.com/JasonHoku/dream-textures/commit/32c78703290d265e6845f31fcc60673053d1b9b3): cancellation callback, ControlNet download availability, and DPT processor API fixes.
- [`63c19fd`](https://github.com/JasonHoku/dream-textures/commit/63c19fda01630034f1df3f11d6b44f6215c9892f): model-folder discovery and preferences interface.
- [`84fdcfc`](https://github.com/JasonHoku/dream-textures/commit/84fdcfcc11893d65c71338a6adc9e54aac87c4d1): projection depth masking, capture diagnostics, and linked ControlNet dispatch.

The compatibility adaptation adds bounded background discovery, conservative checkpoint classification and stable identities, capture cleanup/validation, and focused tests. Jason's packaging workflow also informed the official-actions source-package check. His newer dependency stack and broader planned model/backend migrations were not incorporated. The [review record](compatibility/JASONHOKU_REVIEW_2026-10-20.md) explains each decision. The Dream Textures GPL-3.0-or-later license and existing notices are retained.

## Separately licensed components

The bundled `invisible-watermark` component is by **ShieldMnt**, copyright (c) 2021, under the **MIT License**. This build changes its wheel dependency metadata only. Its original license and exact modification record are retained; see [third-party notices](THIRD_PARTY_NOTICES.md) and [wheel provenance](compatibility/invisible-watermark/).

Runtime libraries and diffusion model weights download separately and retain their own licenses and notices. No diffusion model weights are included in the installation ZIP.

## Repository access and open-source rights

Maintainer control of this GitHub repository is separate from the license rights granted to recipients. Others may use, modify, and redistribute their copies under the applicable license terms; doing so does not grant permission to change this repository.
