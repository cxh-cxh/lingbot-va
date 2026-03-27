# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_aloha_cfg = EasyDict(__name__='Config: VA aloha')
va_aloha_cfg.update(va_shared_cfg)

va_aloha_cfg.wan22_pretrained_model_name_or_path = "/home/rhos/.cache/modelscope/hub/models/Robbyant/lingbot-va-base"
va_aloha_cfg.finetune_model_path = "/mnt/sdb/ckpt/lingbot/turn_on_tap_step_30000"
va_aloha_cfg.attn_window = 72
va_aloha_cfg.frame_chunk_size = 2
va_aloha_cfg.env_type = 'aloha_real'

va_aloha_cfg.height = 256
va_aloha_cfg.width = 256
va_aloha_cfg.action_dim = 30
va_aloha_cfg.action_per_frame = 16
va_aloha_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_aloha_cfg.guidance_scale = 5
va_aloha_cfg.action_guidance_scale = 1

va_aloha_cfg.num_inference_steps = 25
va_aloha_cfg.video_exec_step = -1
va_aloha_cfg.action_num_inference_steps = 50

va_aloha_cfg.snr_shift = 5.0
va_aloha_cfg.action_snr_shift = 1.0

va_aloha_cfg.used_action_channel_ids = list(range(14, 20)) + list(
    range(28, 29)) + list(range(21, 27)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_aloha_cfg.used_action_channel_ids)
] * va_aloha_cfg.action_dim
for i, j in enumerate(va_aloha_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_aloha_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_aloha_cfg.action_norm_method = 'quantiles'
va_aloha_cfg.norm_stat = {
    "q01": 
        [0.] * 14 + [
      -0.13637718558311462,
      0.005198311991989613,
      -1.055068027973175,
      -0.3051827847957611,
      -0.9832136034965515,
      -1.5085920095443726] +
        [0.] +
      [-0.22783608734607697,
      -0.00582629581913352,
      -0.28984948992729187,
      -1.485670566558838,
      -0.7243970036506653,
      -1.4769912958145142
    ] + [0.] + [0.0,0.0],
    "q99": 
        [0.] * 14 + [
      0.2878609001636505,
      2.0274462699890137,
      -0.0008896440267562866,
      1.495966047048569,
      0.4261394739151001,
      0.24735592305660248,] +
        [0.] +
      [0.059135161340236664,
      0.3765985071659088,
      0.0,
      1.5442126989364624,
      0.8024240136146545,
      1.5994752645492554,
    ]+ [0.] + [0.06972000002861023,0.026319999247789383],
}
