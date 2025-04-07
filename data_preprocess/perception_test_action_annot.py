"""
打标action的描述
"""
import copy
import json

import numpy as np
from tqdm import tqdm
import random
random.seed(42)
import math
import os
from typing import List
from call_llm import call_qwen2vl


src_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/'
video_dir = 'perception_test/videos/'
temp_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/action_desc'
tar_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/total_action_desc.json'
tar_dict = {}

for subset in ['train', 'valid']:
    src_file = os.path.join(src_dir, f'all_{subset}.json')

    with open(src_file, 'r') as f:
        src_data = json.load(f)

    for video in tqdm(src_data):
        video_path = os.path.join(video_dir, f'{video}.mp4')
        item = src_data[video]
        video_duration = (item['metadata']['num_frames'] - 1) / item['metadata']['frame_rate']
        real_fps = item['metadata']['frame_rate']
        frame_interval = 1.0 / real_fps

        action_counts = {}

        for action in item["action_localisation"]:
            action_type = action['label'].lower()
            if action_type in ['other', 'clapping hands']:
                continue

            act_id = action['id']
            output_file = os.path.join(temp_dir, f'{video}_{act_id}.txt')

            if os.path.exists(output_file):
                with open(tar_file, 'r', encoding='utf-8') as f:
                    response = f.read()
            else:
                objects = []
                for id_obj in action['parent_objects']:
                    objects.append(item['object_tracking'][id_obj]['label'])

                prompt = f"""
                In this video, the person is {action_type}. The objects related to this action are: {objects}.
                Please replace the pronouns (such as "something", "objects", etc.) in {action_type} \
                with the specific object names from {objects}, while keeping the verb unchanged.
                Please reply directly with the modified result.
                """

                act_start, act_end = action['timestamps']
                act_start, act_end = act_start * 1e-6, act_end * 1e-6
                act_frames = int(math.floor((act_start - act_end) * real_fps))

                video_info = {
                    "type": "video",
                    "video": video_path,
                    "video_start": act_start,
                    "video_end": act_end,
                    "total_pixels": 2 * 24576 * 28 * 28,
                    "min_pixels": 16 * 28 * 2,
                    "max_pixles": 2 * 768 * 28 * 28,
                    "nframes": min(64, act_frames // 2 * 2)
                }

                messages = [
                    {'role': 'user', 'content': [{'type': 'text', 'text': prompt}, video_info]}]
                response = call_qwen2vl(messages, "Qwen2.5-VL-72B-Instruct")
                print(f'{action_type}: {objects}')
                print(response)
                with open(output_file, 'w',encoding='utf-8') as f:
                    f.write(response)


            tar_dict[f'{video}_{act_id}'] = response

with open(tar_file, 'r', encoding='utf-8') as f:
    json.dump(tar_dict, f, ensure_ascii=False, indent=2)