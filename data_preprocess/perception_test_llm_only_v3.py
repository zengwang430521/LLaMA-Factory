"""
只用来训练LLM head
video caption
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


def clear_time_segs(
    time_segs: List[List[int]],
    global_seg: List[int],
    merge: bool = False
) -> List[List[int]]:
    if time_segs is None:
        return time_segs
    if len(time_segs) == 0:
        return []

    time_segs = copy.deepcopy(time_segs)
    global_seg = copy.deepcopy(global_seg)

    global_start, global_end = global_seg

    # 过滤：只剔除完全不相交的段
    filtered = [
        [start, end] for start, end in time_segs
        if not (end < global_start or start > global_end)
    ]

    if not merge or not filtered:
        return sorted(filtered)

    # 合并相交或相连的时间段
    filtered.sort(key=lambda x: x[0])
    merged = [filtered[0]]
    for current in filtered[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # 相交或相连
            last[1] = max(last[1], current[1])  # 合并
        else:
            merged.append(current)

    return merged


def get_frame_label(messages, videos, video_duration, real_fps, mask_history=False):
    video_files = [v['file'] for v in videos]
    video_time_segs = [v['time'] for v in videos]
    valids = []

    for idx, message in enumerate(messages):
        content = copy.deepcopy(message["content"])
        while '<video>' in content:
            if mask_history and idx < len(messages) - 3:
                valids.append(False)
            else:
                valids.append(True)
            content = content.replace("<video>", "", 1)

    assert len(valids) == len(video_time_segs)
    # 先处理一下time_seg
    total_duration = 0
    for i in range(len(video_time_segs)):
        t_start, t_end = video_time_segs[i]
        t_start, t_end = max(t_start, 0), min(t_end, video_duration)
        total_duration += t_end - t_start
        video_time_segs[i] = [t_start, t_end]

    video_fps = 2
    video_maxlen = 64

    frame_nums = []
    for video, time_seg in zip(videos, video_time_segs):
        # 先计算这一段需要采样多少帧
        t_start, t_end = time_seg
        seg_duration = t_end - t_start
        frame_num = min(seg_duration * video_fps, seg_duration * real_fps)
        frame_num = min(frame_num, video_maxlen * seg_duration / total_duration)
        frame_num = math.floor(frame_num)
        frame_num = max(frame_num, 2)  # 最少采集2帧
        if frame_num % 2 != 0:
            # 必须是偶数
            frame_num -= 1
        frame_nums.append(frame_num)

    # 此时各段的采样帧数可能加起来超过 video_maxlen
    current_total = sum(frame_nums)
    # 如果超过，则对各段进行迭代调整，每次从那些帧数大于2的段减少2帧，直到总数不超过总数要求
    while current_total > video_maxlen:
        # import pdb; pdb.set_trace()
        # print('DEBUG: frame index!')

        reduced = False
        for i in range(len(frame_nums)):
            if frame_nums[i] > 2:
                frame_nums[i] -= 2
                current_total -= 2
                reduced = True
                if current_total <= video_maxlen:
                    break
        if not reduced:
            # 如果所有段都已经是2帧，无法再减少，则退出循环
            break

    # 确定采样的frame idx
    frame_times =  []
    for video, time_seg, frame_num in zip(videos, video_time_segs, frame_nums):
        t_start, t_end = time_seg
        sample_times = np.linspace(t_start, t_end, frame_num + 1)[1:]
        frame_times.append(sample_times[1::2])

    # 判断哪些帧需要回答
    frame_labels = []
    for sample_time, video, valid in zip(frame_times, videos, valids):
        frame_label = [-100] * len(sample_time)
        if valid:
            # positive 用闭区间
            positive_time = video.get('positive_time', None)
            if positive_time is not None:
                for t_start, t_end in positive_time:
                    for i, t in enumerate(sample_time):
                        if t_start <= t <= t_end:
                            frame_label[i] = 1

            # negative 用开区间
            negative_time = video.get('negative_time', None)
            if negative_time is not None:
                for t_start, t_end in negative_time:
                    for i, t in enumerate(sample_time):
                        if t_start < t < t_end:
                            frame_label[i] = 0
        frame_labels.append(frame_label)

    return frame_times, frame_labels


# 为了保证没有数据泄漏
test_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/OVO-Bench/ovo_bench.json'
test_videos = set()
with open(test_file, 'r') as f:
    test_data = json.load(f)
for item in test_data:
    if item['task'] == "REC":
        video = item["video"]
        if 'perception_test' in video:
            test_videos.add(os.path.basename(video).split('.mp4')[0])


instructions = [
    {"role": "user", "content": "Please concisely narrate the video in real time."},
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
tar_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/processed/REC_trainval_llm_only_v3.json'


tar_data = []
label_count = {0: 0, 1: 0, -100: 0}
num_other = 0

act_desc_file= '/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/total_action_desc.json'
with open(act_desc_file, 'r', encoding='utf-8') as f:
    act_desc_dict = json.load(f)

for subset in ['train', 'valid']:
    src_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/all_{subset}.json'

    with open(src_file, 'r') as f:
        src_data = json.load(f)

    for video in tqdm(src_data):
        if video in test_videos:
            print(f'skip:{subset}/{video}.mp4')
            continue

        video_path = f'perception_test/videos/{video}.mp4'
        item = src_data[video]
        video_duration = (item['metadata']['num_frames'] - 1) / item['metadata']['frame_rate']
        real_fps = item['metadata']['frame_rate']
        frame_interval = 1.0 / real_fps

        messages, videos = [], []

        query = random.choice(instructions)['content']
        query_time = 0
        messages.append({"role": "user", "content": query, 'ignore_end_stream': True})

        last_time = query_time

        for action in item["action_localisation"]:
            act_id = action['id']
            action_type = action['label']
            act_desc = action_type.lower()

            if act_desc == 'other':
                continue

            if act_desc != 'clapping hands':
                act_desc = act_desc_dict[f'{video}_{act_id}']

            answer = f'The person is {act_desc}.'
            answer = answer.replace('..', '.')

            act_start, act_end = action['timestamps'][0] * 1e-6, action['timestamps'][1] * 1e-6
            if isinstance(answer_insert_point, list) or isinstance(answer_insert_point, tuple):
                insert_point = random.uniform(answer_insert_point[0], answer_insert_point[1])
            else:
                insert_point = answer_insert_point
            response_time = act_start + insert_point * (act_end - act_start)

            video_info = {
                "file": video_path,
                "time": [last_time, response_time],
            }
            messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True})
            videos.append(video_info)
            messages.append({"role": "assistant", "content": answer, "valid": True})
            last_time = response_time

        if messages[-1]['role'] == 'user':
            t = 0
            continue
        tar_item = {"messages": messages, "videos": videos}
        tar_data.append(tar_item)

        frame_times, frame_labels = get_frame_label(copy.deepcopy(messages), videos, video_duration, real_fps, mask_history=False)
        for frame_label in frame_labels:
            for l in frame_label:
                label_count[l] += 1
        t = 0

print(label_count)
print(len(tar_data))

os.makedirs(os.path.dirname(tar_file), exist_ok=True)
with open(tar_file, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)


# {0: 0, 1: 0, -100: 148575} 7859


