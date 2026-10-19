import bpy
prompt = bpy.context.scene.dream_textures_prompt
backend = prompt.get_backend()

prompt.steps = 20
prompt.cfg_scale = 7.5
prompt.scheduler = 'DPM Solver Multistep'
prompt.step_preview_mode = 'Accurate'
backend.sdp_attention = True
backend.attention_slicing = True
backend.attention_slice_size_src = 'auto'
backend.attention_slice_size = 1
backend.cudnn_benchmark = False
backend.tf32 = False
backend.half_precision = True
backend.cpu_offload = 'off'
backend.channels_last_memory_format = False
backend.batch_size = 1
backend.vae_slicing = True
backend.cpu_only = False
