"""
视频： [......视频a......] [......视频b......][......目标片段1......][......视频c......][......目标片段2......][...视频d...]
文字：                   Q                           Answer1                                 Answer2
"""

import json
import os
from os.path import join, exists
import random
from tqdm import tqdm
from os.path import join
from utils import get_frame_label
import copy


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


answer_insert_point = [0.7, 1.0]

src_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/"
tar_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed/v5"
os.makedirs(tar_dir, exist_ok=True)
src_path = join(src_dir, 'dvc_train-human_anno-0.25_0.5_earlier.json')
tar_path = join(tar_dir, "dvc_train-human_anno-0.25_0.5_earlier_v5_llm.json")

with open(src_path, 'r') as f:
    src_data = json.load(f)

tar_data = []
label_count = {0: 0, 1: 0, -100: 0}

for video_file in tqdm(src_data.keys()):
    for _, conversations in src_data[video_file].items():
        valid = True
        video_path = join('shot2story-videos', video_file)
        instruction = random.choice(instructions)['content']
        video_duration = conversations[-1]['timespan'][1]

        messages, videos = [], []
        # 加入instruction
        query_time = 0
        if query_time > 0:
            messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, 'valid': True})
            videos.append({"file": video_path, "time": [0, query_time]})
        messages.append({"role": "user", "content": instruction, 'ignore_end_stream': True, 'valid': True})
        last_time = 0

        for idx in range(len(conversations)):
            src_conv = conversations[idx]
            src_start, src_end = src_conv['timespan']
            response = src_conv["text"]

            # 确定回答插入的位置
            if isinstance(answer_insert_point, list) or isinstance(answer_insert_point, tuple):
                insert_point = random.uniform(answer_insert_point[0], answer_insert_point[1])
            else:
                insert_point = answer_insert_point

            response_time = src_start + insert_point * (src_end - src_start)
            if response_time <= last_time:
                print('error: 时间间隔太小，放弃')
                valid = False
                break

            video_info = {
                "file": video_path,
                "time": [last_time, response_time],
                "positive_time": [[-2.0, -1.0]],
                "negative_time": [[-2.0, -1.0]],
            }
            messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, "valid": True})
            videos.append(video_info)
            messages.append({"role": "assistant", "content": response, 'ignore_end_stream': True, "valid": True})
            last_time = response_time

        tar_item = {"messages": copy.deepcopy(messages), "videos": copy.deepcopy(videos)}
        tar_data.append(tar_item)

with open(tar_path, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)
print(len(tar_data))

