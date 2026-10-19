# Contributing

This is an experimental Dream Textures compatibility port for Blender 5.2.2 and the Windows NVIDIA runtime. Start with the [build report](../compatibility/BUILD_REPORT.md) and [reproducible test instructions](../compatibility/TESTING.md).

## Preserve the runtime boundary

- Keep the installed module name **`dream_textures`** so registration and existing preferences remain compatible.
- Install AI dependencies through the private runtime installer. Keep Blender's own Python packages and other applications' environments separate.
- Update exact dependency pins and validation together. CUDA validation is required for Windows runtime activation; the CPU lane is for development tests.
- Preserve validation before activation, restart-based runtime selection, cancellation, and rollback behavior.

## Propose and verify changes

Describe the problem, the resulting behavior, and the relevant checks in each pull request. Run the tests that exercise the changed behavior; use the packaging instructions when preparing an installable ZIP. State the platform and hardware actually tested, and distinguish CPU execution from Windows GPU validation.

For a bug report, include Blender and add-on versions, operating system, reproduction steps, and the relevant error or installation log excerpt. Include the GPU and driver details for CUDA failures. Remove credentials from any shared logs.

Retain upstream attribution, license notices, and third-party provenance. Source changes remain **GPL-3.0-or-later**; see [LICENSE](../LICENSE) and [third-party notices](../THIRD_PARTY_NOTICES.md).
