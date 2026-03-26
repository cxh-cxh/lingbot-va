# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_aloha_cfg import va_aloha_cfg
import os

va_aloha_train_cfg = EasyDict(__name__='Config: VA aloha train')
va_aloha_train_cfg.update(va_aloha_cfg)

# va_aloha_train_cfg.resume_from = '/robby/share/Robotics/lilin1/code/Wan_VA_Release/train_out/checkpoints/checkpoint_step_10'

va_aloha_train_cfg.dataset_path = '/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/worldmodelgroup/lingbot-va/data/turn_on_tap'
va_aloha_train_cfg.empty_emb_path = os.path.join(va_aloha_train_cfg.dataset_path, 'empty_emb.pt')
va_aloha_train_cfg.enable_wandb = False
va_aloha_train_cfg.load_worker = 16
va_aloha_train_cfg.save_interval = 1000
va_aloha_train_cfg.gc_interval = 50
va_aloha_train_cfg.cfg_prob = 0.1

# Training parameters
va_aloha_train_cfg.learning_rate = 1e-5
va_aloha_train_cfg.beta1 = 0.9
va_aloha_train_cfg.beta2 = 0.95
va_aloha_train_cfg.weight_decay = 0.1
va_aloha_train_cfg.warmup_steps = 10
va_aloha_train_cfg.batch_size = 1 
va_aloha_train_cfg.gradient_accumulation_steps = 1
va_aloha_train_cfg.num_steps = 50000 
