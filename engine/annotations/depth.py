import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
import threading
from .compat import UNIFORM_COLOR
from ...projection_utils import normalize_depth

def render_depth_map(context, collection=None, invert=True, width=None, height=None, matrix=None, projection_matrix=None, main_thread=False):
    e = threading.Event()
    result = None
    failure = None

    width, height = width or context.scene.render.resolution_x, height or context.scene.render.resolution_y
    matrix = matrix or context.scene.camera.matrix_world.inverted()
    projection_matrix = projection_matrix or context.scene.camera.calc_matrix_camera(
        context,
        x=width,
        y=height
    )

    def _execute():
        nonlocal result, failure
        offscreen = None
        previous_depth_test = previous_depth_mask = None
        try:
            previous_depth_test = gpu.state.depth_test_get()
            previous_depth_mask = gpu.state.depth_mask_get()
            offscreen = gpu.types.GPUOffScreen(width, height)
            with offscreen.bind():
                fb = gpu.state.active_framebuffer_get()
                fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1)
                gpu.state.depth_test_set('LESS_EQUAL')
                gpu.state.depth_mask_set(True)
                with gpu.matrix.push_pop():
                    gpu.matrix.load_matrix(matrix)
                    gpu.matrix.load_projection_matrix(projection_matrix)
                    shader = gpu.shader.from_builtin(UNIFORM_COLOR)
                    shader.bind()
                    # Adapted from Jason Hoku's 84fdcfcc11893d65c71338a6adc9e54aac87c4d1:
                    # alpha is the geometry mask, so the uniform must be explicit.
                    shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))

                    def render_mesh(mesh, transform):
                        mesh.transform(transform)
                        mesh.calc_loop_triangles()
                        vertices = np.empty((len(mesh.vertices), 3), 'f')
                        indices = np.empty((len(mesh.loop_triangles), 3), 'i')
                        mesh.vertices.foreach_get("co", vertices.ravel())
                        mesh.loop_triangles.foreach_get("vertices", indices.ravel())
                        batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
                        batch.draw(shader)

                    if collection is None:
                        objects = ((instance.object, instance.matrix_world, False)
                                   for instance in context.object_instances)
                    else:
                        objects = ((obj, obj.matrix_world, True) for obj in collection.objects)
                    for obj, transform, from_collection in objects:
                        try:
                            mesh = obj.to_mesh(depsgraph=context) if from_collection else obj.to_mesh()
                            try:
                                if mesh is not None:
                                    render_mesh(mesh, transform)
                            finally:
                                obj.to_mesh_clear()
                        except Exception:
                            # Non-mesh/unsupported scene objects do not contribute depth.
                            continue
                depth = np.asarray(fb.read_depth(0, 0, width, height).to_list())
                alpha = np.asarray(fb.read_color(0, 0, width, height, 4, 0, 'UBYTE').to_list())[:, :, 3]
                result = normalize_depth(depth, alpha > 0, invert=invert)
        except BaseException as error:
            failure = error
        finally:
            cleanups = []
            if previous_depth_test is not None:
                cleanups.append(lambda: gpu.state.depth_test_set(previous_depth_test))
            if previous_depth_mask is not None:
                cleanups.append(lambda: gpu.state.depth_mask_set(previous_depth_mask))
            if offscreen is not None:
                cleanups.append(offscreen.free)
            for cleanup in cleanups:
                try:
                    cleanup()
                except BaseException as error:
                    if failure is None:
                        failure = error
            # Timer failures must also release a waiting worker and propagate.
            e.set()
    if main_thread or threading.current_thread() == threading.main_thread():
        _execute()
    else:
        bpy.app.timers.register(_execute, first_interval=0)
        e.wait()
    if failure is not None:
        raise failure
    return result
