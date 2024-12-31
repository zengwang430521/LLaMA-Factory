"""
视频： [......视频a......] [......视频b......][......目标片段1......][......视频c......][......目标片段2......][...视频d...]
文字：                   Q                           Answer1                                 Answer2

data:
[......视频a......] Query [......视频b......][......目标片段1......][......视频c......][......目标片段2......][...视频d...] Answer1
------------------------n nnnnnnnnnnnnnnnnn ----------------YYYYY YYYY-----nnnnnnnnn -----------------YYYY YYYY----nnn -------

[......视频a......] Query [......视频b......][......目标片段1 Answer1 ......][......视频c......][......目标片段2......][...视频d...] Answer2
------------------------- ----------------- ----------------------- nnnnnn nnnnnnnnnnnnnnnnnn -----------------YYYY YYYY---nnnn -------

[......视频a......] Query [......视频b......][......目标片段1 Answer1 ......][......视频c......][......目标片段2 Answer2......][...视频d...] Fake Answer
------------------------ ------------------ ----------------------- ------ ------------------ ----------------------nnnnnnn nnnnnnnnnnnn -----------

完全的负样本
"""

import copy
import json

import numpy as np
from tqdm import tqdm
import random
random.seed(42)
import math
import os
from collections import defaultdict


def get_frame_label(messages, video_duration, real_fps, mask_history=True):
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


query_template = """ 
In the video, the man/woman is {activity} repetitively. 
Your task is to count how many times he/she has completed the action of {activity}.
Remind me every time when he/she finishes one.
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
"""

# 为了保证没有数据泄漏
# test_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/OVO-Bench/ovo_bench.json'
test_file = '/afs/zengwang/projects/task_define_service/OVO-Bench/data/ovo_bench.json'
test_videos = set()
with open(test_file, 'r') as f:
    test_data = json.load(f)
for item in test_data:
    if item['task'] == "REC":
        video = item["video"]
        if 'perception_test' in video:
            test_videos.add(os.path.basename(video).split('.mp4')[0])


# session_per_video = 1
# query_point = [0, 0.3]
# # tar_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/processed/REC_trainval_stream_only_v5_neg_1.json'
# tar_file = f'/afs/zengwang/projects/task_define_service/data/processed/perception_test/REC_trainval_stream_only_v5_neg_1.json'


session_per_video = 4
query_point = [0, 0.3]
tar_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/processed/REC_trainval_stream_only_v5_neg_2.json'
tar_file = f'/afs/zengwang/projects/task_define_service/data/processed/perception_test/REC_trainval_stream_only_v5_neg_2.json'


# 统计 action 出现的次数：
action_freq = defaultdict(lambda: 0)
for subset in ['train', 'valid']:
    # src_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/all_{subset}.json'
    src_file = f'/afs/zengwang/projects/task_define_service/data/perception_test/all_{subset}.json'

    with open(src_file, 'r') as f:
        src_data = json.load(f)
    for video in tqdm(src_data):
        if video in test_videos:
            print(f'skip:{subset}/{video}.mp4')
            continue
        video_path = f'perception_test/videos/{video}.mp4'
        item = src_data[video]
        for action in item['action_localisation']:
            action_freq[action['label']] += 1


def sample_actions(action_freq, n, action_except):
    # 过滤掉 action_out 中的 key
    filtered_actions = {k: v for k, v in action_freq.items() if k not in action_except}

    # 如果过滤后的字典为空或样本数量大于可用键数，返回空列表
    if not filtered_actions:
        return []

    if len(filtered_actions) <= n:
        return list(filtered_actions.keys())

    # 计算总频次
    total_freq = sum(filtered_actions.values())

    # 计算每个键的概率
    action_prob = {k: v / total_freq for k, v in filtered_actions.items()}

    # 按照概率进行不放回采样
    sampled_actions = random.choices(list(filtered_actions.keys()), weights=list(action_prob.values()), k=n)
    return sampled_actions


tar_data = []
label_count = {0: 0, 1: 0, -100: 0}

for subset in ['train', 'valid']:
    # src_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/all_{subset}.json'
    src_file = f'/afs/zengwang/projects/task_define_service/data/perception_test/all_{subset}.json'

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

        action_counts = {}
        for action in item["action_localisation"]:
            action_type = action['label']
            if action_type in action_counts:
                action_counts[action_type] += 1
            else:
                action_counts[action_type] = 1

        neg_actions = sample_actions(action_freq, session_per_video, ['Other', 'other'] + list(action_counts.keys()))

        for activity in neg_actions:
            query = query_template.format(activity=activity.lower())

            if isinstance(query_point, list) or isinstance(query_point, tuple):
                q_point = random.uniform(query_point[0], query_point[1])
            else:
                q_point = query_point
            query_time = int(math.floor(q_point * video_duration))
            messages, videos = [], []
            if query_time > 0:
                messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, "valid": True})
                videos.append({"file": video_path, "time": [0, query_time], "positive_time": [[-2.0, -1.0]], "negative_time": [[-2.0, -1.0]]})
            messages.append({"role": "user", "content": query, 'ignore_end_stream': False, "valid": True})

            # 用来训练保持沉默
            # fake answer 不能用来训练lm_head
            video_info = {
                "file": video_path,
                "time": [query_time, video_duration],
                "positive_time": [[-2.0, -1.0]],
                "negative_time": [[query_time-frame_interval, video_duration+frame_interval]]
            }
            messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, "valid": True})
            videos.append(video_info)
            messages.append({"role": "assistant", "content": '', 'ignore_end_stream': True, "valid": False})

            tar_item = {"messages": copy.deepcopy(messages), "videos": copy.deepcopy(videos)}
            tar_data.append(tar_item)
            frame_times0, frame_labels0 = get_frame_label(copy.deepcopy(messages), video_duration, real_fps, mask_history=True)
            frame_times, frame_labels = get_frame_label(copy.deepcopy(messages), video_duration, real_fps, mask_history=False)
            assert frame_labels0 == frame_labels
            for frame_label in frame_labels:
                for l in frame_label:
                    label_count[l] += 1
            t = 0

print(tar_file.split('stream_only_')[-1], label_count, len(tar_data))

os.makedirs(os.path.dirname(tar_file), exist_ok=True)
with open(tar_file, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)


# v5_neg_1.json {0: 155460, 1: 0, -100: 22828} 8044
# v5_neg_2.json {0: 621840, 1: 0, -100: 91281} 32176