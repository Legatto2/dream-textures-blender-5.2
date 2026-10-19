# Compatibility build report

## Result and scope

**Dream Textures 0.4.2-compat.1** was built on 19 October 2026 for **Blender 5.2.2 LTS / Windows x64 / Python 3.13 / RTX 5060**.

The implementation passes **47 automated tests**, and the production dependency set resolves to Windows Python 3.13 wheels. The add-on was exercised in the official Blender 5.2.2 Linux application and its embedded Python. A Windows GPU installation and a pretrained model generation are the remaining target-machine acceptance checks.

The tests establish API/runtime compatibility. They do not establish pretrained output quality, performance, available VRAM in a particular scene, every ControlNet auxiliary checkpoint, all interactive painting/projection/viewport behavior, or compatibility with other operating systems.

## Baseline and runtime

| Item | Value |
|---|---|
| Upstream add-on | Dream Textures 0.4.1 |
| Upstream commit | c2622a8a9f1ae6b790cfe1d2571f814b126811b4 |
| Compatibility build | 0.4.2-compat.1; Blender displays tuple version 0.4.2 |
| Test application | Blender 5.2.2 LTS, d13f752e3b9c |
| Embedded test interpreter | Python 3.13.13 |
| Blender frontend NumPy | 2.3.4, retained from Blender |
| Private backend NumPy | 2.3.5 |
| Windows Torch lane | 2.9.1+cu128; torchvision 0.24.1+cu128 |
| CPU validation lane | 2.9.1+cpu; torchvision 0.24.1+cpu |
| Diffusion integration | diffusers 0.27.2 |
| Transformers / tokenizer | 4.48.3 / 0.21.1 |
| Hugging Face Hub / accelerate | 0.25.2 / 1.1.1 |

The official Blender archive SHA-256 was verified against its published checksum file. The Linux environment initially produced four incomplete files during GNU tar extraction. Those members were restored directly from the checksum-verified archive with Python streaming extraction. The restored Blender executable and embedded interpreter then ran successfully. Tests used disposable configuration/cache paths. Local execution outside the network-isolated sandbox was used for Blender application tests because the sandbox's PulseAudio IPC restriction blocked application shutdown; test inputs and model execution remained offline.

### Dependency decisions

Diffusers 0.27.2 preserves Dream Textures' custom depth and tiling integration. Hugging Face Hub 0.25.2 preserves its required cached_download API. Python 3.13-compatible native packages and a Blackwell-capable Torch lane replace the old interpreter-specific runtime.

The 52 exact production pins are recorded in the two compatibility requirements files. A final binary-only resolver check selected Windows x64/Python 3.13 or compatible pure/ABI3 wheels. A corresponding CPU runtime was installed and matched all 52 CPU-lane pins. The Windows resolution is a packaging check, not a Windows execution test.

Only opencv-python-headless supplies cv2. The small local invisible-watermark wheel changes its OpenCV dependency metadata while preserving every original code/model/license byte. See the wheel provenance and MIT notice.

Unused historical OmegaConf, Lightning, and TensorBoard requirements were omitted after examining the conversion paths. PyYAML and safetensors remain. Arbitrary legacy pickled checkpoints may still need migration; the public default model uses FP16 safetensors.

## Meaningful verification

| Group | Result | Evidence |
|---|---:|---|
| Blender frontend | 11 passed | evidence/blender-frontend-smoke.log |
| Backend inference and image operations | 14 passed | evidence/backend-smoke.log |
| Private installer invariants | 16 passed | evidence/installer-tests.log |
| Actual Blender/worker integration | 6 passed | evidence/actor-integration.log |
| Windows wheel resolution | 52 packages resolved | evidence/final-windows-resolution.json |
| CPU runtime versus pins | 0 mismatches | evidence/cpu-runtime-requirements-check.json |
| Exact isolated runtime validation command | Passed | evidence/runtime-check.json |
| Default model download selection | Passed; metadata only | evidence/first-run-download-selection.json |

Frontend tests cover cold add-on enable, cleanup and re-enable, Cycles hook restoration, node interfaces containing panels, default pipeline creation, NumPy/image orientation and alpha, sRGB render conversion, render error propagation, current brush properties, safe missing-runtime polls, and the bundled presets. GPU drawing and actual interactive paint strokes were not part of this headless test.

Backend tests perform actual Torch/diffusers inference with tiny randomly initialized models. They cover reproducible text-to-image, image-to-image, nine-channel inpainting, depth with guidance 7.5 and 1, ControlNet, SDXL, fourfold upscaling, seamless padding, tiled VAE/accurate previews, safetensors/configuration handling, Canny/Shuffle preprocessing, watermark encode/decode, unknown-length downloads, and eight grayscale dtype/shape/mode combinations. No pretrained diffusion weights were downloaded for testing.

Worker tests confirm that Blender keeps its own NumPy while its worker imports private NumPy/Torch with the embedded interpreter. Tests cover NumPy transfer, a third-party exception, cancellation, a deliberate worker crash, and an unreadable dependency-specific return value. Each failure completes its future and the next request recovers.

Installer tests cover one isolated resolver command, interpreter discovery, environment sanitization, cancellation of owned child processes, activation after validation, rollback, previous-runtime retention, ABI/path checks, DLL search-directory handles, process-lifetime selection caching, and lock release.

The packaging gate **passed** using Blender's actual add-on ZIP installer. It enabled the installed copy, confirmed version 0.4.2 and the Model offload default, and disabled it with clean RNA/Cycles restoration. See evidence/zip-install-smoke.log. Documentation and this log were included afterward; every production file is checked against the tested copy in evidence/tested-production-hashes.json.

## Runtime boundary and lifecycle

The frontend uses Blender's existing Python environment. Its worker imports only the validated private runtime plus Blender's standard library and the add-on's source. Python and virtual-environment startup variables are cleared for spawning and restored immediately in the frontend.

The runtime is installed below Blender's configuration directory:

- dream_textures_compat/runtime.json — selected/previous runtime IDs.
- dream_textures_compat/runtimes/ — isolated completed or staging runtimes.
- dream_textures_compat/logs/ — install and validation logs.
- dream_textures_compat/jobs/ — progress, cancellation, and validation records.

Installation uses Blender's bundled ensurepip wheel directly, with isolated pip execution, fixed indexes, pinned binary wheels, and a new private target. Validation occurs in a fresh subprocess. Activation is an atomic pointer change after successful checks; an old validated runtime is retained.

The running Blender process caches its original runtime selection, including when the add-on is toggled. A restart loads the selected new runtime. No loaded DLL directory is renamed during activation.

Automatic checks for old upstream binary releases are disabled for this compatibility build. The original project link remains available.

## Target-machine acceptance

On Windows, the installer verifies native/library imports, CUDA availability, CUDA 12.8, the architecture list for an RTX 50-series device, and small FP16 matrix multiplication/convolution operations. A failure leaves the previous runtime selected.

After installation and a full Blender restart, the settings should show **NVIDIA GeForce RTX 5060**, Torch 2.9.1+cu128, and CUDA 12.8. The user then downloads or links a model and performs a 512 × 512, batch-1 generation. A resulting image confirms the basic Windows/GPU path.

The default public SD 1.5 repository was checked anonymously at revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14. The real download-selection code accepted a blank token and selected 15 files totaling 2,742,217,630 advertised bytes. FP16 safetensors were selected and .bin weights excluded. The snapshot download was intercepted before downloading weights.

The Blender UI now defaults CPU Offload to Model for the 8 GB target. That trades some transfer time for lower peak GPU residency. Actual performance has not been measured on the user's RTX 5060.

## Included source and provenance

The ZIP supplies all modified implementation files, existing-file patch, exact requirements, test source, build source, test logs, resolver report, model metadata evidence, wheel provenance, and license notices. Compiled Python caches and the old dependency placeholder are excluded.
