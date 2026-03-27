## IMPORTANT
Some CPU models do not support BF16. The server script will check it and change the precision of offloaded modules.

## How to run remote inference (on ALOHA)
1. Set the `finetune_model_path` in `wan_va/configs/va_aloha_cfg.py`. If using base model, set it to `None`.
2. Run `pytorch script/set_attn_mode.py aloha --mode flashattn` to change the `attn_mode` config or do it manually.
3. Set the host IP in `wan_va/configs/shared_config.py` and `evaluation/aloha/launch_client.sh`
4. Run `bash evaluation/aloha/launch_server.sh` on the server and run `bash evaluation/aloha/launch_client.sh` on ALOHA.

## TODO
1. Add video recording.
2. Improve inference speed.