"""Projection input helpers that do not import Blender or the AI runtime.

Depth masking and flat/empty capture handling adapt Jason Hoku's work in
JasonHoku/dream-textures commit 84fdcfcc11893d65c71338a6adc9e54aac87c4d1.
The viewport workflow and generation size remain unchanged in this backport.
"""

from contextlib import contextmanager

import numpy as np


def normalize_depth(depth, geometry_mask, *, invert=True):
    """Normalize finite geometry pixels; keep background zero in either mode."""
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(geometry_mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Depth and its geometry mask must have matching 2D shapes.")
    covered = mask & np.isfinite(depth)
    result = np.zeros(depth.shape, dtype=np.float32)
    if not np.any(covered):
        return result
    values = depth[covered]
    low, high = float(values.min()), float(values.max())
    if high - low <= 1e-8:
        result[covered] = 0.5
    else:
        values = (values - low) / (high - low)
        result[covered] = np.clip(1.0 - values if invert else values, 0.0, 1.0)
    return result


def validate_depth_capture(depth):
    depth = np.asarray(depth)
    if depth.ndim != 2 or not depth.size or not np.all(np.isfinite(depth)):
        raise ValueError("The viewport depth capture is invalid.")
    if not np.any(depth > 0):
        raise ValueError(
            "Depth capture is empty: no geometry was captured from this viewport. "
            "Check the Dream Textures Projection Depth image."
        )


def validate_color_capture(image, size):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("The viewport color capture must contain RGB or RGBA pixels.")
    actual = (image.shape[1], image.shape[0])
    if actual != tuple(size):
        raise ValueError(
            f"Color capture size {actual[0]} x {actual[1]} does not match "
            f"depth capture size {size[0]} x {size[1]}."
        )
    if not np.all(np.isfinite(image)):
        raise ValueError("The viewport color capture contains invalid pixel values.")


def projection_task(task_api, *, depth, image, strength, use_control_net):
    """Choose the projection task independently of hidden init-image settings."""
    if use_control_net:
        if image is not None:
            return task_api.ImageToImage(image=image, strength=strength, fit=True)
        return task_api.PromptToImage()
    return task_api.DepthToImage(depth=depth, image=image, strength=strength)


@contextmanager
def temporary_render_settings(render, space, size, filepath):
    """Restore capture settings and saved format-dependent options on every exit."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("The projection viewport must have a positive size.")
    image_settings = render.image_settings
    saved = [
        (render, name, getattr(render, name))
        for name in ("resolution_x", "resolution_y", "resolution_percentage", "filepath")
    ]
    # Blender 5 restricts file_format by media_type. Both setters can coerce
    # color/depth, multiview, and output color space; restore them in that order.
    if hasattr(image_settings, "media_type"):
        saved.append((image_settings, "media_type", image_settings.media_type))
    saved.append((image_settings, "file_format", image_settings.file_format))
    for name in ("color_mode", "color_depth", "views_format"):
        if hasattr(image_settings, name):
            saved.append((image_settings, name, getattr(image_settings, name)))
    colorspace = getattr(image_settings, "linear_colorspace_settings", None)
    if colorspace is not None and hasattr(colorspace, "name"):
        saved.append((colorspace, "name", colorspace.name))
    for name in ("use_border", "use_crop_to_border"):
        if hasattr(render, name):
            saved.append((render, name, getattr(render, name)))
    saved.append((space.overlay, "show_overlays", space.overlay.show_overlays))
    if hasattr(space, "show_gizmo"):
        saved.append((space, "show_gizmo", space.show_gizmo))
    try:
        render.resolution_x, render.resolution_y = width, height
        render.resolution_percentage = 100
        render.filepath = filepath
        if hasattr(image_settings, "media_type"):
            image_settings.media_type = 'IMAGE'
        image_settings.file_format = 'PNG'
        if hasattr(image_settings, "color_mode"):
            image_settings.color_mode = 'RGBA'
        for name in ("use_border", "use_crop_to_border"):
            if hasattr(render, name):
                setattr(render, name, False)
        space.overlay.show_overlays = False
        if hasattr(space, "show_gizmo"):
            space.show_gizmo = False
        yield
    finally:
        failures = []
        for owner, name, value in saved:
            try:
                setattr(owner, name, value)
            except Exception as error:
                failures.append((name, error))
        if failures:
            names = ", ".join(name for name, _ in failures)
            raise RuntimeError(f"Could not restore projection capture settings: {names}") from failures[0][1]
