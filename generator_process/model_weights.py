"""Check availability of the unsharded ControlNet weights this backend supports.

Adapted from JasonHoku/dream-textures commit 32c7870's incomplete-download check.
The classifier and loader share exact filenames and require a non-empty regular
file. This checks availability only; it does not validate tensor contents.
"""

import os
import stat


CONTROLNET_FP16_WEIGHTS = (
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.fp16.bin",
)
CONTROLNET_FP32_WEIGHTS = (
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
)


def find_controlnet_weights(snapshot_folder, half_precision=False):
    """Return an available weight filename, preserving precision preference."""
    names = (
        CONTROLNET_FP16_WEIGHTS + CONTROLNET_FP32_WEIGHTS
        if half_precision else CONTROLNET_FP32_WEIGHTS + CONTROLNET_FP16_WEIGHTS
    )
    for name in names:
        try:
            # Hub snapshots can link to cached blobs; inspect the target file.
            details = os.stat(os.path.join(snapshot_folder, name))
        except OSError:
            continue
        if stat.S_ISREG(details.st_mode) and details.st_size > 0:
            return name
    return None
