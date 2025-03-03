import json
from tqdm import tqdm
import random
import math
import os

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

src_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/COIN/annotations/COIN.json'
tar_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/COIN/annotations/processed/SSR_COIN_choice.json'

with open(src_file, 'r') as f:
    src_data = json.load(f)


tar_data = []
for video_id in tqdm(src_data['database'].keys()):
    if video_id in test_videos:
        print(f'skip: {video_id}.mp4')
        continue

    item = src_data['database'][video_id]
    video_path = f"COIN/videos/{item['recipe_type']}/{video_id}.mp4"
    if not os.path.exists(f"/home/SENSETIME/zengwang/myprojects/task_define_service/data/COIN/annotations/videos/{item['recipe_type']}/{video_id}.mp4"):
        print(f'skip: {video_id}')
        continue

    tutorial = item["class"]
    all_steps = item["annotation"]
    formatted_steps = '; '.join(f'{chr(65 + i)}. {step["label"]}' for i, step in enumerate(all_steps)) + ';'
    query = query_template.format(tutorial=tutorial, formatted_steps=formatted_steps)

    first_response_time = all_steps[0]['segment'][0]

    max_query_time = math.floor(first_response_time - 1)
    max_query_time = max(max_query_time, 0)
    query_time = float(random.randint(0, int(max_query_time)))

    messages, videos = [], []
    if query_time > 0:
        messages.append({"role": "user", "content": "<video>", "time": [0.0, query_time]})
        videos.append(video_path)
    messages.append({"role": "user", "content": query, "time": [query_time, query_time]})

    last_time = query_time
    for idx, step in enumerate(all_steps, start=0):
        answer = f'{chr(65 + idx)}'
        response_time = min(step['segment'][0] + 1,  step['segment'][1])

        messages.append({"role": "user", "content": "<video>", "time": [last_time, response_time]})
        videos.append(video_path)
        messages.append({"role": "assistant", "content": answer, "time": [response_time, response_time]})
        last_time = response_time

    tar_item = {"messages": messages, "videos": videos}
    tar_data.append(tar_item)

    t = 0

os.makedirs(os.path.dirname(tar_file), exist_ok=True)
with open(tar_file, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False)
print(len(tar_data))