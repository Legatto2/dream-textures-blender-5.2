# JasonHoku fork review — 20 October 2026

## Source reviewed

The supplied [Package Release workflow run](https://github.com/JasonHoku/dream-textures/actions/runs/29301740046) completed successfully on **14 July 2026** at commit [`bf477cde6223f45bf3ed4cf5d1a3cc168fe38231`](https://github.com/JasonHoku/dream-textures/commit/bf477cde6223f45bf3ed4cf5d1a3cc168fe38231). It packaged Windows CUDA dependencies for Python 3.11 and 3.13. Its job steps performed installation, archiving, and artifact upload; that run does not establish Blender 5.2.2 or RTX 5060 inference acceptance.

The review also includes Jason's later `main` snapshot [`50d53f0724f94f1c8449ed749a711de4cd2991e4`](https://github.com/JasonHoku/dream-textures/commit/50d53f0724f94f1c8449ed749a711de4cd2991e4), four commits after the supplied run. Across the common upstream baseline [`c2622a8a9f1ae6b790cfe1d2571f814b126811b4`](https://github.com/carson-katri/dream-textures/commit/c2622a8a9f1ae6b790cfe1d2571f814b126811b4), that snapshot adds nine commits touching 18 files. The substantive application changes are in three commits:

- [`32c7870`](https://github.com/JasonHoku/dream-textures/commit/32c78703290d265e6845f31fcc60673053d1b9b3): newer dependency API port, worker/download fixes, and loader changes.
- [`63c19fd`](https://github.com/JasonHoku/dream-textures/commit/63c19fda01630034f1df3f11d6b44f6215c9892f): model-folder discovery.
- [`84fdcfc`](https://github.com/JasonHoku/dream-textures/commit/84fdcfcc11893d65c71338a6adc9e54aac87c4d1): projection, depth capture, and ControlNet UI/dispatch fixes.

## Integration decisions

| Area | Decision for this fork |
|---|---|
| Default cancellation callback | Adapt the Python binding fix so a new Future can report its uncancelled state and still accept the actor's per-instance callback. |
| Partial ControlNet downloads | Adapt availability checks using loader-supported non-empty weight files, and provide an explanatory error. File availability does not verify weight contents. |
| DPT image processing | Replace the deprecated DPTFeatureExtractor alias with DPTImageProcessor, which is already present in our pinned Transformers 4.48.3. |
| Linked ControlNet files | Resolve linked checkpoint identifiers in Inpaint and ImageToImage as well as PromptToImage. |
| Local model folders | Adapt the explicit folder/rescan interface with bounded header parsing, conservative supported-model discovery, stable path identifiers, malformed-file handling, and background scanning. Preserve manual checkpoint linking and its selected configurations. |
| Model menu persistence | Extend the adaptation to save new model/ControlNet/refiner selections by full identifier. Blender menu positions must not silently choose another checkpoint after a rescan. Preserve legacy numeric interpretation without counting new folder rows, and resolve a saved folder identity when that file is explicitly linked. |
| Depth capture | Adapt the explicit opaque shader mask and finite empty/flat-depth handling. Apply the geometry mask consistently for both depth directions. |
| Projection diagnostics and cleanup | Retain inspectable add-on-owned captures, restore changed render/GPU state on failure, and validate captures before changing materials or UVs. Diagnostic captures are not promised to be the final resized model tensors. |
| Depth and Color | Explicitly select the corresponding image task and use consistent top-to-bottom image orientation. |
| Packaging and checks | Use official actions pinned to commits for focused tests and source-package validation. Keep the private runtime installer rather than bundling multi-gigabyte GPU dependencies. |

## Changes retained from our fork

The worker already has private-runtime selection, interpreter/environment isolation, serialized spawning, dead-worker detection, one-shot nonblocking cancellation transport, and recovery after failed responses. Replacing it with Jason's worker would discard those protections.

The current cached-offload repair also remains in place. It removes tracked and recursive hooks before cache pruning and ControlNet replacement. Jason's alternative checks only a legacy final hook later in conversion and does not supersede the tested lifecycle fix. Unknown/zero download totals and iteration over cached model values were already corrected here.

## Features and migrations left separate

**Modern model support is still planned in Jason's source.** His [modernization document](https://github.com/JasonHoku/dream-textures/blob/50d53f0724f94f1c8449ed749a711de4cd2991e4/docs/MODERNIZATION-2026.md) labels the runtime port as Phase 1, with Flux, Z-Image, GGUF/quantization, architecture-aware generation, and related work in Phase 2. A ComfyUI generation backend is Phase 3. Discovering compatible files inside a ComfyUI model folder does not run ComfyUI workflows.

His new requirements use Torch 2.13/CUDA 13 and Diffusers 0.39, with several broader dependency ranges. Our tested stack stays on Torch 2.9.1/CUDA 12.8, Diffusers 0.27.2, Transformers 4.48.3, Hub 0.25.2, and Accelerate 1.1.1. The newer single-file/config/import APIs cannot be copied directly into that pinned runtime. Existing SDXL code paths remain distinct from the manually validated SD1.5 path; this review does not add a new claim of target-machine SDXL acceptance.

Camera and aspect-fit projection are useful future options, but require separate live alignment checks. The current backport preserves viewport selection and generation sizing. Jason's Camera default would change the existing workflow, and clamping both fitted axes to at least 512 can exceed a 512-square pixel budget. His SDXL-only ControlNet recommendation is also unsuitable for our default SD1.5 test configuration.

The folder scanner was adapted rather than copied as-is: malformed tensor records could abort refresh, unknown/component-only or pickle files could be offered as usable checkpoints, display labels could collide, and unbounded scans could block Blender. Automatic discovery is restricted to confidently recognized supported Safetensors checkpoints. Unknown or ambiguous files continue through explicit manual linking.

The selection integration uses Blender's documented enum transform callbacks to retain access to old internally stored numeric values while saving new choices as separate full strings. The source contract was checked against [Blender 5.2.2's property examples](https://github.com/blender/blender/blob/v5.2.2/doc/python_api/examples/bpy.props.5.py); the transform accessors are also documented in the declared minimum [Blender 5.2.0](https://github.com/blender/blender/blob/v5.2.0/doc/python_api/examples/bpy.props.5.py). Session enum numbers are not reused, and dynamic enum strings are retained. Getter callbacks do not write a migration into older files; selecting a model saves the full identifier.

VRAM peaks alone do not establish driver spillover or a particular slowdown. No such diagnosis or performance claim is added.

## Attribution and validation

Jason Hoku is credited for the reviewed and adapted work, alongside Carson Katri and the original Dream Textures contributors. Source remains under the existing GPL-3.0-or-later terms; see [ATTRIBUTION.md](../ATTRIBUTION.md) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The [Windows validation record](VALIDATION_2026-10-20.md) describes **compat.2 plus the original two-file hotfix**. The additional backports require their own focused checks and subsequent interactive Blender acceptance; the earlier three-run result must not be relabeled as a test of these newly integrated features. Reproduction commands and current automated results are recorded in [TESTING.md](TESTING.md).
