import os,shutil
import json

data_dir = 'data/turn_on_tap'
if os.path.exists(os.path.join(data_dir,"meta","episodes_ori.jsonl")):
    print("Skip")
    exit(0)
    
shutil.move(os.path.join(data_dir,"meta","episodes.jsonl"),os.path.join(data_dir,"meta","episodes_ori.jsonl"))

with open(os.path.join(data_dir,"meta","episodes_ori.jsonl"),'r') as f_in:
    with open(os.path.join(data_dir,"meta","episodes.jsonl"),'w') as f_out:
        for line in f_in:
            if line.strip():
                data = json.loads(line)
                data["action_config"] = [{"start_frame": 0, "end_frame": data["length"], "action_text": data["tasks"][0], "skill": ""}]
                f_out.write(json.dumps(data) + '\n')