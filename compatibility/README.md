# Compatibility development notes

User setup and troubleshooting are in the [Blender 5.2 guide](../docs/BLENDER_5_2.md). This directory holds implementation notes, validation records, test instructions, and build evidence for the compatibility fork.

| Document | Contents |
|---|---|
| [Revision notes](https://github.com/Legatto2/dream-textures-blender-5.2/blob/blender-5.2-compat/.github/RELEASE_NOTES.md) | Changes by build and package checksums. |
| [Build report](BUILD_REPORT.md) | Compatibility work, runtime design, and historical build results. |
| [Testing and packaging](TESTING.md) | Reproduction commands, suite scope, and how to create an installation ZIP. |
| [Windows validation — 20 October 2026](VALIDATION_2026-10-20.md) | Reported hardware results, settings, evidence limits, and remaining acceptance work. |
| [JasonHoku fork review](JASONHOKU_REVIEW_2026-10-20.md) | Source provenance, adapted improvements, and integration decisions. |
| [Evidence](evidence/) | Retained logs and supporting build records. |
| [Build manifest](BUILD_MANIFEST.json) | File sizes and hashes for the packaged source. |
| [Watermark component](invisible-watermark/) | Bundled wheel provenance and license information. |

Validation records distinguish reported Windows GPU use, focused offline checks, and historical Blender/runtime execution. See each record for the build and environment it covers.

For development conventions and contributions, see [CONTRIBUTING.md](https://github.com/Legatto2/dream-textures-blender-5.2/blob/blender-5.2-compat/.github/CONTRIBUTING.md).
