# Windows generation and ControlNet validation — 20 October 2026

## Result

**Basic pretrained image generation and one-face ControlNet projection work on the reported Windows setup.** After applying the two-file hotfix to compat.2 and restarting Blender, Michael Heald reported **three consecutive projection runs without errors**, with the generator left running between runs. **Seamless was Auto** during those three runs.

Source revision **0.4.2-compat.3** integrates those fixes and additional backports from Jason Hoku's fork. It remains experimental and unreleased. The hardware check used **compat.2 plus the original two-file hotfix**, before the additional backports or a fresh installation of a compat.3 package.

## Evidence and scope

The Windows results below are based on the maintainer's reports and screenshots reviewed during the 19–20 October troubleshooting session. They are separate from automated execution in the development environment. Raw desktop screenshots and the full Windows log are not committed here.

| Check | Result and limit |
|---|---|
| Windows runtime and pretrained SD 1.5 generation | Reported working on Windows 11, Blender 5.2.2 LTS, NVIDIA GeForce RTX 5060. This establishes the basic generation path on that machine. |
| Repeated Depth ControlNet projection | Three consecutive runs without reported errors after the hotfix. The generator stayed running. Seamless was Auto. This is a targeted regression check, not a soak test. |
| One selected cube face | Later screenshots show the generated brick pattern on the selected face using the projected material and UV map. This establishes basic projection, not every UV layout or every face. |
| Pale strip along the face's top | The enlarged generated image contains a light gray cap directly above the brick panel. That content explains the corresponding strip in the projection; the image also contains surrounding sky. |
| Recommended-model preference UI | The stale field name is corrected and covered by a focused simulated validation-UI test. The specific recommendation UI path was not separately reported as retested in Windows. |
| JasonHoku backports | Local model folders, projection capture/diagnostics, Depth and Color routing, and other backend fixes have focused offline checks. They were added after the reported Windows runs and still need interactive Blender acceptance. |
| New compat.3 installation package | Source packaging checks can establish archive integrity; installation of the newly assembled package in Blender remains unverified. |
| Baking and broader workflows | Bake was off for this projection check. Baking, reprojection onto existing UV layouts, interactive painting, and broader viewport/render workflows are still unverified on the target machine. |
| Performance and other platforms | No controlled latency/VRAM benchmark, extended soak test, or other-machine acceptance result was obtained. |

## Reported test configuration

| Setting | Value |
|---|---|
| Base build used on Windows | 0.4.2-compat.2 with the 20 October two-file hotfix |
| Blender / operating system / GPU | 5.2.2 LTS / Windows 11 / NVIDIA GeForce RTX 5060 |
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| ControlNet | `lllyasviel/control_v11f1p_sd15_depth` |
| Projection inputs | Depth; Use ControlNet enabled; scale 1.0 |
| Generation | 512 × 512; 20 steps; guidance 7.5; batch 1 |
| Memory settings | Half Precision enabled; CPU Offload Model; SDP Attention enabled |
| Bake | Off |
| Seamless during the three-run check | Auto, as clarified by the maintainer |

A later framing check used a close Top Orthographic view with the face framed within the viewport. Seamless Off and a fixed seed of 42 were recommended for that separate content comparison; the record does not reclassify the earlier three-run test as Off or claim every recommended setting was independently verified. Image variation alone is not a generation failure. Random Seed and Seamless control different behavior.

## Failure and correction

The Windows traceback showed a successful denoising run followed by later attempts failing at the start of ControlNet execution. The relevant error was a CUDA input reaching a linear layer whose weights were on CPU:

```text
RuntimeError: Expected all tensors to be on the same device,
but got mat1 is on cuda:0, different from other tensors on cpu
```

The cached loader reuses model components while rebuilding the ControlNet container and constructing a new pipeline through `AutoPipeline.from_pipe()`. That conversion shares modules but does not transfer all pipeline hook bookkeeping. Existing component hooks can make the optimizer regard offload as already configured while a replacement ControlNet container has no corresponding hook.

The fix removes the old pipeline's tracked hooks and recursively detaches component hooks **before** cache pruning or ControlNet replacement. The optimizer can then install a complete set on the reconstructed pipeline while retaining the cached weights. The separate UI correction changes the recommended-model download field from `prefer_fp16_revision` to the existing `prefer_fp16_variant` property.

The following hashes identify the **original delivered two-file hotfix tested on compat.2**. They are historical evidence, not hashes of the current files: subsequent backports modify ControlNet weight selection and backend model/dispatch code. The `_remove_cached_pipeline_hooks` helper and `load_model()` cache lifecycle function remain exactly unchanged from the integrated hotfix. Current packaged file hashes are recorded in [BUILD_MANIFEST.json](BUILD_MANIFEST.json).

| File | SHA-256 |
|---|---|
| `generator_process/actions/load_model.py` | `857bbb20342059c8a13b9cf10ea4802ac0f5e3352bc62eb39a788bdd1bbdb814` |
| `diffusers_backend.py` | `2f2f210622a22eee0fba16e7136bd4482d6daf5846c5b32d70550e1075cddf6f` |

Relevant source: [loader](../generator_process/actions/load_model.py), [optimizer](../generator_process/models/optimizations.py), [projection](../operators/project.py), [ControlNet generation](../generator_process/actions/control_net.py), and [validation UI](../diffusers_backend.py). Dependency behavior was checked against [Diffusers 0.27.2 AutoPipeline](https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/pipelines/auto_pipeline.py) and [Accelerate 1.1.1 hooks](https://github.com/huggingface/accelerate/blob/v1.1.1/src/accelerate/hooks.py).

## Automated checks for this source update

The focused standard-library suite, [test_offload_lifecycle.py](tests/test_offload_lifecycle.py), exercises the actual loader and optimizer against simulated Diffusers, Accelerate, and device behavior. It covers cached repeated calls, changed ControlNet containers, cleanup order, nested sequential hooks, cache retention, an offload-disabled control, and the corrected recommendation UI property. It does **not** run neural inference, Blender, Torch, or CUDA.

The integrated suite passes **8 tests**. Against the original pre-hotfix snapshot, it fails as expected and reproduces the simulated device mismatch on attempts two and three. A mutation that disables recursive detachment is also rejected by the strengthened nested-hook checks. Test commands and the captured current-run output are linked from [TESTING.md](TESTING.md).

The existing private installer suite was also rerun: **16 tests passed**. These initial two suites ran on Linux with **Python 3.12.14**, using standard-library simulations and temporary files; they did not install or execute a production GPU runtime. See [offload regression output](evidence/offload-lifecycle-2026-10-20.log) and [installer output](evidence/installer-2026-10-20.log).

The combined run passed **90 tests**: 16 installer, 8 offload/UI, 8 backend backport, 20 projection input, 19 model discovery, 5 model dispatch/refiner, and 14 selection-persistence tests. It used Linux Python 3.12.14 and NumPy 2.3.5. The [combined offline output](evidence/offline-backports-2026-10-20.log) and [suite descriptions](TESTING.md#current-offline-compatibility-checks) record the actual execution. All **108 Python files** also passed source compilation. NumPy is used for array operations; Blender/GPU/model dependencies remain simulated. These checks do not transfer the earlier Windows acceptance to the new features. The [fork review](JASONHOKU_REVIEW_2026-10-20.md) records exactly what was adapted and what remains separate.

The historical **47 application/runtime tests** belong to compat.1: 11 Blender frontend, 14 real tiny-model CPU backend, 16 private installer, and 6 Blender worker tests. Their existing logs and production hashes are preserved. That full suite was not rerun for compat.3, and its count must not be presented as a new Windows or hotfix test result.

## What remains

1. Install the newly assembled compat.3 package in Blender and confirm it selects the existing validated runtime.
2. Check the new Model Folders UI, model selection after rescan, and the Depth / Depth and Color capture diagnostics in Blender. Confirm selected models/configurations and capture orientation using the target setup.
3. Exercise Bake and projection onto an existing UV layout, then broader interactive workflows and representative production assets.
4. Measure performance and run longer sessions on the target hardware before making stability or throughput claims.

The brick cap and surrounding sky are generated image content. Improving that output is a texture-quality task; it is not evidence that the repaired repeat-run cache lifecycle is failing.
