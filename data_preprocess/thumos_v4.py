# Thanks for submitting the access form.
# The passwords of THUMOS15 and 14 are "THUMOS15_challenge_REGISTERED" and "THUMOS14_REGISTERED", respectively.

import json
import os
from os.path import join
from collections import defaultdict
from tqdm import tqdm
import math
import random


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


tar_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/thumos/processed/REC_valid.json'
tar_data = []

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

            if version == '15_valid':
                video_path = join(f'thumos/videos/{video_id}.mp4')
            elif version == '14_valid':
                video_path = join(f'thumos/validation/{video_id}.mp4')

            actions = action_dict[video_id]

            # 只使用多次出现的动作
            if len(actions) < 2:
                continue

            filtered_videos_ids.add(video_id)

            actions = sorted(actions, key=lambda x: x[0])
            first_response_time = actions[0][1]

            max_query_time = math.floor(first_response_time - 1)
            max_query_time = max(max_query_time, 0)
            query_time = float(random.randint(0, int(max_query_time)))

            messages, videos = [], []
            if query_time > 0:
                messages.append({"role": "user", "content": "<video>", "time": [0.0, query_time]})
                videos.append(video_path)
            messages.append({"role": "user", "content": query, "time": [query_time, query_time]})

            last_time = query_time
            for count, action in enumerate(actions, start=1):
                answer = str(count)
                # response_time = action[1]
                '''
                response_period: 表示可以进行回复的区间 [t1, t2, t3, t4]
                0:  不回复
                1:  回复
                -:  不监督
                ......t1 ...... t2 ...... t3 ......t4.......
                000000----------111111111111---------0000000
                '''
                t_start, t_end = action
                delta = t_end - t_start
                response_period = [t_start + 0.4 * delta, t_start + 0.6 * delta, t_end, t_end + 1]
                response_time = t_end  # 为了充分训练，插入response的位置要尽量靠后一些

                messages.append({"role": "user", "content": "<video>", "time": [last_time, response_time]})
                videos.append(video_path)
                messages.append({"role": "assistant", "content": answer, "time": response_period})
                last_time = response_time

            tar_item = {"messages": messages, "videos": videos}
            tar_data.append(tar_item)

os.makedirs(os.path.dirname(tar_file), exist_ok=True)
with open(tar_file, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)
print(len(tar_data))


# with open('/home/SENSETIME/zengwang/myprojects/task_define_service/data/thumos/processed/filtered_videos_ids.json', 'w', encoding='utf-8') as f:
#     json.dump(list(filtered_videos_ids), f, ensure_ascii=False, indent=2)
# print(len(filtered_videos_ids))

