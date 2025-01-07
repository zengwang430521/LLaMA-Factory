import json
import os
from os.path import join, exists
import random
from tqdm import tqdm


instructions = [
    {"role": "user", "content": "Please concisely narrate the video in real time."},
    {"role": "user", "content": "Help me to illustrate my view in short."},
    {"role": "user", "content": "Please simply describe what do you see."},
    {"role": "user", "content": "Continuously answer what you observed with simple text."},
    {"role": "user", "content": "Do concise real-time narration."},
    {"role": "user", "content": "Hey assistant, do you know the current video content? Reply me concisely."},
    {"role": "user", "content": "Simply interpret the scene for me."},
    {"role": "user", "content": "What can you tell me about? Be concise."},
    {"role": "user", "content": "Use simple text to explain what is shown in front of me."},
    {"role": "user", "content": "What is the action now? Please response in short."},
]


src_path = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/dvc_train-human_anno-0.25_0.5_earlier.json'
tar_path = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed/dvc_train-human_anno-0.25_0.5_earlier_sample10.json'
os.makedirs(os.path.dirname(tar_path), exist_ok=True)

with open(src_path, 'r') as f:
    src_data = json.load(f)

tar_data = []
n = -1
for video_file in tqdm(src_data.keys()):
    n += 1
    if n % 10 != 0:
        continue
    for _, conversations in src_data[video_file].items():
        instruction = random.choice(instructions)
        text_historys = []
        text_historys.append({"role": "user", "content": instruction["content"], "time": [0.0, 0.0]})
        for src_conv in conversations:
            response_time = src_conv['timespan'][1]
            video_message = {"role": "user", "content": "<video>", "time": [0.0, response_time]}
            response_message = {"role": "assistant", "content": src_conv["text"], "time": [response_time, response_time]}
            tar_item = {
                "messages": text_historys + [video_message, response_message],
                "videos": [video_file]
            }
            tar_data.append(tar_item)
            text_historys.append(response_message)

with open(tar_path, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)









t = 0