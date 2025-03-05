import json
import sys

from tqdm import tqdm
import random
import math
import os

query_template = """ 
In the video, the man/woman is {activity} repetitively. 
Your task is to count how many times he/she has completed the action of {activity}.
Remind me every time when he/she finishes one.
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
"""

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

tar_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/processed/REC_trainval_ovo_test.json'
tar_data = []
idx = 1
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
        action_counts = {}
        for action in item["action_localisation"]:
            action_type = action['label']
            if action_type in action_counts:
                action_counts[action_type] += 1
            else:
                action_counts[action_type] = 1

        for activity in action_counts.keys():
            # 只利用出现了多次的动作生成计数数据
            if action_counts[activity] < 2:
                continue
            filtered_action = [action for action in item["action_localisation"] if action['label'] == activity]

            start_times, end_times, test_infos = [], [], []
            test_infos.append({"realtime": filtered_action[0]['timestamps'][1]*0.5*1e-6, "count": 0})
            for count, action in enumerate(filtered_action, start=1):
                start_time, end_time = action['timestamps']
                start_time, end_time = start_time * 1e-6, end_time * 1e-6

                start_times.append(start_time)
                end_times.append(end_time)
                test_infos.append({"realtime": end_time, "count": count})
            test_infos.append({"realtime": end_time + 1, "count": count})

            test_data = {
                "id": idx,
                "task": "REC",
                "video": f"data/perception_test_videos/{video}.mp4",
                "activity": activity.lower(),
                "start_times": start_times,
                "end_times": end_times,
                "test_info": test_infos
            }
            tar_data.append(test_data)
            if len(tar_data) >= 50:
                os.makedirs(os.path.dirname(tar_file), exist_ok=True)
                with open(tar_file, 'w', encoding='utf-8') as f:
                    json.dump(tar_data, f, ensure_ascii=False, indent=2)
                print(len(tar_data))

                sys.exit(0)