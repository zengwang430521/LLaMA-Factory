# Thanks for submitting the access form.
# The passwords of THUMOS15 and 14 are "THUMOS15_challenge_REGISTERED" and "THUMOS14_REGISTERED", respectively.

from collections import defaultdict
import math
import json
import os
import random
from tqdm import tqdm
from os.path import join
from utils import get_frame_label
import copy


query_template = """ 
In the video, the man/woman is {activity} repetitively. 
Your task is to count how many times he/she has completed the action of {activity}.
Remind me every time when he/she finishes one.
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
"""

test_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/OVO-Bench/ovo_bench.json'
test_videos = set()
with open(test_file, 'r') as f:
    test_data = json.load(f)
for item in test_data:
    if item['task'] == "REC":
        video = item["video"]
        if 'thumos' in video:
            test_videos.add(os.path.basename(video).split('.mp4')[0])




min_pos_duration = None
answer_insert_point = 0.3
stream_positive_point = 0.7
action_extend_time = 1
action_bridge_time = 1
tar_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/thumos/processed/REC_valid_v5_llm.json'

video_info_file = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/thumos_video_info.json"
with open(video_info_file, 'r', encoding='utf-8') as f:
    video_info_dict = json.load(f)


tar_data = []
label_count = {0: 0, 1: 0, -100: 0}
filtered_videos_ids = set()
for version in ['14_valid', '15_valid']:
    if version == '15_valid':
        src_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/thumos/TH15_Temporal_annotations_validation/annotations'
    elif version == '14_valid':
        src_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/thumos/TH14_Temporal_annotations_validation/annotation'
    elif version == '14_test':
        src_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/thumos/TH14_Temporal_Annotations_Test/annotations/annotation'
    else:
        raise ValueError(str(version))

    for filename in tqdm(sorted(os.listdir(src_dir))):
        src_file = join(src_dir, filename)
        if version == '15_valid':
            activity = filename.split('.txt')[0].split('_')[-1].lower()
        elif version in ['14_valid', '14_test']:
            activity = filename.split('.txt')[0].split('_')[0].lower()
        else:
            raise ValueError(str(version))

        if activity == 'ambiguous':
            continue
        print(activity)
        query = query_template.format(activity=activity)

        with open(src_file, 'r') as f:
            lines = f.readlines()
        action_dict = defaultdict(list)
        for line in lines:
            video_id, t_start, t_end = line.split()
            action_dict[video_id].append([float(t_start), float(t_end)])

        for video_id in action_dict.keys():
            if video_id in test_videos:
                print(f'skip: {video_id}')
                continue

            if version == '15_valid':
                video_path = join(f'thumos/videos/{video_id}.mp4')
            elif version == '14_valid':
                video_path = join(f'thumos/validation/{video_id}.mp4')

            actions = action_dict[video_id]
            # # 只使用多次出现的动作
            # if len(actions) < 2:
            #     continue

            video_k = join('data', video_path)
            if video_k in video_info_dict.keys():
                video_info = video_info_dict[video_k]
                video_duration = video_info['duration']
            else:
                print(f'No video: {video_k}')
                continue

            filtered_videos_ids.add(video_id)

            actions = sorted(actions, key=lambda x: x[0])
            max_query_time = actions[0][0]
            max_query_time = math.floor(max_query_time - 1)
            max_query_time = max(max_query_time, 0)
            query_time = float(random.randint(0, int(max_query_time)))

            messages, videos = [], []

            if query_time > 0:
                messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, "valid": True})
                videos.append({"file": video_path, "time": [0, query_time], "positive_time": [[-2.0, -1.0]], "negative_time": [[-2.0, -1.0]]})
            messages.append({"role": "user", "content": query, 'ignore_end_stream': False, "valid": True})
            last_time = query_time
            for idx in range(len(actions)):
                count = idx + 1
                answer = str(count)

                src_start, src_end = actions[idx]
                src_start = max(src_start, query_time)

                # 确定回答插入的位置
                if isinstance(answer_insert_point, list) or isinstance(answer_insert_point, tuple):
                    insert_point = random.uniform(answer_insert_point[0], answer_insert_point[1])
                else:
                    insert_point = answer_insert_point
                response_time = src_start + insert_point * (src_end - src_start)
                video_info = {
                    "file": video_path,
                    "time": [last_time, response_time],
                    "positive_time": [[-2.0, -1.0]],
                    "negative_time": [[-2.0, -1.0]],
                }
                messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, "valid": True})
                videos.append(video_info)
                # 监督 llm loss, valid = True
                messages.append({"role": "assistant", "content": answer, 'ignore_end_stream': True, "valid": True})
                last_time = response_time

            tar_item = {"messages": copy.deepcopy(messages), "videos": copy.deepcopy(videos)}
            tar_data.append(tar_item)

os.makedirs(os.path.dirname(tar_file), exist_ok=True)
with open(tar_file, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)
print(len(tar_data), label_count)



