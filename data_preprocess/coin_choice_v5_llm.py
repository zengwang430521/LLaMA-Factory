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
"""

import json
import os
import random
from tqdm import tqdm
from os.path import join
from utils import get_frame_label
import copy
import math


query_template = """
    You're watching a tutorial video of {tutorial}. It contains the following steps:
    {formatted_steps}
    Your task is to tell me what step the video is at.
    Remind me every time when he/she turns to a new step.
    Respond only with the letter corresponding to current step (e.g., A, B, C). 
    Do not include any additional text or explanation in your response.
"""

# 为了保证没有数据泄漏
test_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/OVO-Bench/ovo_bench.json'
test_videos = set()
with open(test_file, 'r') as f:
    test_data = json.load(f)
for item in test_data:
    if item['task'] == "SSR":
        video = item["video"]
        if 'COIN' in video:
            test_videos.add(os.path.basename(video).split('.mp4')[0])


min_pos_duration = None
answer_insert_point = 0.3
stream_positive_point = 0.7
action_extend_time = 1
action_bridge_time = 1

src_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/COIN/annotations/COIN.json'
tar_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/COIN/annotations/processed/SSR_COIN_choice_v5_llm.json'

with open(src_file, 'r') as f:
    src_data = json.load(f)


tar_data = []
label_count = {0: 0, 1: 0, -100: 0}

for video_id in tqdm(src_data['database'].keys()):
    if video_id in test_videos:
        print(f'skip: {video_id}.mp4')
        continue

    item = src_data['database'][video_id]
    video_path = f"COIN/videos/{item['recipe_type']}/{video_id}.mp4"
    if not os.path.exists(f"/home/SENSETIME/zengwang/myprojects/task_define_service/data/COIN/annotations/videos/{item['recipe_type']}/{video_id}.mp4"):
        # print(f'skip: {video_id}')
        continue
    video_duration = max(item['duration'], item['end'])

    tutorial = item["class"]
    all_steps = item["annotation"]

    # formatted_steps = '; '.join(f'{chr(65 + i)}. {step["label"]}' for i, step in enumerate(all_steps)) + ';'

    # 随机生成A, B, C, D等选项顺序
    option_labels = [chr(65 + i) for i in range(len(all_steps))]
    shuffled_labels = option_labels.copy()
    random.shuffle(shuffled_labels)
    label_step_map = dict(zip(shuffled_labels, all_steps))
    formatted_steps = '; '.join(f'{label}. {label_step_map[label]["label"]}' for label in sorted(label_step_map)) + ';'

    query = query_template.format(tutorial=tutorial, formatted_steps=formatted_steps)

    max_query_time = all_steps[0]['segment'][0]
    max_query_time = math.floor(max_query_time - 3)
    max_query_time = max(max_query_time, 0)
    query_time = float(random.randint(0, int(max_query_time)))

    messages, videos = [], []
    if query_time > 0:
        messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, 'valid': True})
        videos.append({"file": video_path, "time": [0, query_time]})
    messages.append({"role": "user", "content": query, 'ignore_end_stream': False, 'valid': True})
    last_time = query_time

    for idx, step in enumerate(all_steps, start=0):
        # answer = f'{chr(65 + idx)}'
        # 找出当前 step 对应的 label（反向查找）
        for label, mapped_step in label_step_map.items():
            if mapped_step == step:
                answer = label
                break
        src_start, src_end = step['segment']
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
    json.dump(tar_data, f, ensure_ascii=False)
print(len(tar_data))
print(label_count, len(tar_data))
# 5765
