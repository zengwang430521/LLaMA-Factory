import copy
import json
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


tar_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/processed/REC_trainval_stream_only.json'
tar_data = []

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
        video_duration = item['metadata']['num_frames'] / item['metadata']['frame_rate']

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

            query = query_template.format(activity=activity.lower())
            first_response_time = filtered_action[0]['timestamps'][0] * 1e-6

            max_query_time = math.floor(first_response_time - 1)
            max_query_time = max(max_query_time, 0)
            query_time = float(random.randint(0, int(max_query_time)))

            messages, videos = [], []
            if query_time > 0:
                messages.append({"role": "user", "content": "<video>", "time": [0.0, query_time]})
                videos.append(video_path)
            messages.append({"role": "user", "content": query, "time": [query_time, query_time]})

            last_time = query_time
            for count, action in enumerate(filtered_action, start=1):
                answer = str(count)
                # response_time = action['timestamps'][1] * 1e-6

                '''
                response_period: 表示可以进行回复的区间 [t1, t2, t3, t4]
                0:  不回复
                1:  回复
                -:  不监督
                ......t1 ...... t2 ...... t3 ......t4.......
                000000----------111111111111---------0000000
                '''
                t_start, t_end = action['timestamps'][0] * 1e-6, action['timestamps'][1] * 1e-6

                """
                这一轮的监督数据, 把回答放在最后，只监督stream label
                stream label: 动作前: 0, 动作中: --, 动作后: 1
                """
                response_period = [t_start, t_end, video_duration, video_duration+1]
                response_time = video_duration

                messages.append({"role": "user", "content": "<video>", "time": [last_time, response_time]})
                videos.append(video_path)
                messages.append({"role": "assistant", "content": answer, "time": response_period})

                # 这里必须deepcopy，因为后续需要修改messages
                tar_item = {"messages": copy.deepcopy(messages), "videos": copy.deepcopy(videos)}
                tar_data.append(tar_item)

                """
                先删除之前加入的特殊数据
                然后把这一轮的回复插在正常的位置（t_end)，便于后一轮的监督
                同时让这一轮的 stream label 都是 无监督-
                """
                messages = messages[:-2]
                videos = videos[:-1]

                response_period = [0, video_duration, video_duration+1, video_duration+2]
                response_time = t_end

                messages.append({"role": "user", "content": "<video>", "time": [last_time, response_time]})
                videos.append(video_path)
                messages.append({"role": "assistant", "content": answer, "time": response_period})

                last_time = response_time


os.makedirs(os.path.dirname(tar_file), exist_ok=True)
with open(tar_file, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)
print(len(tar_data))