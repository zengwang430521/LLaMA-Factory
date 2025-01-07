import json
import os
from os.path import join, exists
import random
from tqdm import tqdm




src_path = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/magqa_train-0.25_0.5-earlier.json'
tar_path = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed/magqa_train-0.25_0.5-earlier.json'
os.makedirs(os.path.dirname(tar_path), exist_ok=True)

with open(src_path, 'r') as f:
    src_data = json.load(f)

tar_data = []
for src_item in tqdm(src_data):
    video_start_time = src_item["video_start_time"]
    if video_start_time > 0:
        print(video_start_time)

    video_file = src_item["video_uid"]
    conversations = src_item["conversation"]
    instruction = conversations[0]

    text_historys = []
    text_historys.append({"role": "user", "content": instruction["content"], "time": [instruction["time"], instruction["time"]]})
    for src_conv in conversations[1:]:
        response_time = src_conv['timespan'][1]
        video_message = {"role": "user", "content": "<video>", "time": [video_start_time, response_time]}
        response_message = {"role": "assistant", "content": src_conv["content"], "time": [response_time, response_time]}
        tar_item = {
            "messages": text_historys + [video_message, response_message],
            "videos": [video_file]
        }
        tar_data.append(tar_item)
        text_historys.append(response_message)

with open(tar_path, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)









t = 0