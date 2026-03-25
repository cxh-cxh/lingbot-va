import json
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from wan_va.configs import VA_CONFIGS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="flex"
    )
    
    args = parser.parse_args()
    
    config_path = Path(VA_CONFIGS[args.config].wan22_pretrained_model_name_or_path) / "transformer" / "config.json"
    
    with open(config_path,'r') as f:
        config = json.load(f)
    
    config["attn_mode"] = args.mode
    print(f"Set {config_path} attn_mode to {args.mode}")
    
    with open(config_path,'w') as f:
        json.dump(config,f,indent=4)
    
if __name__ == "__main__":
    main()