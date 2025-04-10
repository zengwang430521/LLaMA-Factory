"""
打标action的描述
"""
from call_llm import call_qwen2vl, _read_video_decord_v2
import qwen_vl_utils
qwen_vl_utils.vision_process.VIDEO_READER_BACKENDS['decord'] = _read_video_decord_v2

import copy
import json

import numpy as np
from tqdm import tqdm
import random
random.seed(42)
import math
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed


# src_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/'
# video_dir = 'perception_test/videos/'
# temp_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/action_desc'
# tar_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/total_action_desc.json'
# tar_dict = {}


src_dir = '/afs/zengwang/projects/task_define_service/data/perception_test'
video_dir = '/afs/zengwang/projects/task_define_service/data/perception_test/videos/'
temp_dir = '/afs/zengwang/projects/task_define_service/data/perception_test/action_desc'
tar_file = '/afs/zengwang/projects/task_define_service/data/perception_test/total_action_desc.json'
tar_dict = {}

os.makedirs(temp_dir, exist_ok=True)


MAX_WORKERS = 200  # 可修改最大并行数



def process_action(video: str, action: dict, item: dict, video_path: str, real_fps: float) -> (str, str):
    action_type = action['label'].lower()
    if action_type in ['other', 'clapping hands']:
        return None, None

    act_id = action['id']
    output_file = os.path.join(temp_dir, f'{video}_{act_id}.txt')
    if f'{video}_{act_id}' == 'video_10993_0':
        import pdb; pdb.set_trace()
        print(f'{video}_{act_id}')

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            response = f.read()
    else:
        objects = [item['object_tracking'][oid]['label'] for oid in action['parent_objects']]
        prompt = f"""
        In this video, the person is {action_type}. The objects related to this action are: {objects}.
        Please replace the pronouns (such as "something", "objects", etc.) in {action_type} \
        with the specific object names from {objects}, while keeping the verb unchanged.

        For example: 
        Input: opening something, ['box']
        Result: opening a box. 

        Please reply directly with the modified action, such as opening a box.
        Do not include any additional text or explanation in your response.
        """

        act_start, act_end = action['timestamps']
        act_start, act_end = act_start * 1e-6, act_end * 1e-6
        act_frames = int(math.floor((act_end - act_start) * real_fps))

        # video_info = {
        #     "type": "video",
        #     "video": video_path,
        #     "video_start": act_start,
        #     "video_end": act_end,
        #     "total_pixels": 2 * 24576 * 28 * 28,
        #     "min_pixels": 16 * 28 * 2,
        #     "max_pixles": 2 * 768 * 28 * 28,
        #     "nframes": max(min(32, act_frames // 2 * 2), 2)
        # }

        video_info = {
            "type": "video",
            "video": video_path,
            "video_start": act_start,
            "video_end": act_end,
            "total_pixels": 2 * 12288 * 28 * 28,
            "min_pixels": 16 * 28 * 2,
            "max_pixles": 2 * 384 * 28 * 28,
            "nframes": max(min(16, act_frames // 2 * 2), 2)
        }

        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, video_info]}]
        response = call_qwen2vl(messages, "Qwen2.5-VL-72B-Instruct")

        print('-'*100)
        print(f'{action_type}: {objects}')
        print(response)
        print('-'*100)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response)

    return f'{video}_{act_id}', response


for subset in ['train', 'valid']:
    src_file = os.path.join(src_dir, f'all_{subset}.json')
    with open(src_file, 'r') as f:
        src_data = json.load(f)

    # for video, item in src_data.items():
    #     video_path = os.path.join(video_dir, f'{video}.mp4')
    #     real_fps = item['metadata']['frame_rate']
    #     for action in item["action_localisation"]:
    #         process_action(video, action, item, video_path, real_fps)

    tasks = []
    for video, item in src_data.items():
        video_path = os.path.join(video_dir, f'{video}.mp4')
        real_fps = item['metadata']['frame_rate']
        for action in item["action_localisation"]:
            tasks.append((video, action, item, video_path, real_fps))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_action, *args) for args in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {subset}"):
            result = future.result()
            if result and result[0]:
                tar_dict[result[0]] = result[1]

with open(tar_file, 'w', encoding='utf-8') as f:
    json.dump(tar_dict, f, ensure_ascii=False, indent=2)