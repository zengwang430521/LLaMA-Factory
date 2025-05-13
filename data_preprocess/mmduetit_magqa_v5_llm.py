import json
import os
import random
from tqdm import tqdm
from os.path import join
from utils import get_frame_label
import copy


min_pos_duration = None
answer_insert_point = 0.3
stream_positive_point = 0.7
action_extend_time = 1
action_bridge_time = 1


src_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/"
tar_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed/v5"
os.makedirs(tar_dir, exist_ok=True)
src_path = join(src_dir, 'magqa_train-0.25_0.5-earlier.json')
tar_path = join(tar_dir, "magqa_train-0.25_0.5-earlier_v5_llm.json")

# 先获取视频长度
video_duration_dict = {}
tmp_path = join(src_dir, 'dvc_train-human_anno-0.25_0.5_earlier.json')
with open(tmp_path, 'r') as f:
    tmp_data = json.load(f)
for video_file in tqdm(tmp_data.keys()):
    for _, conversations in tmp_data[video_file].items():
        video_duration = conversations[-1]['timespan'][1]
        if video_file in video_duration_dict.keys():
            video_duration_dict[video_file] = max(video_duration_dict[video_file], video_duration)
        else:
            video_duration_dict[video_file] = video_duration

with open(src_path, 'r') as f:
    src_data = json.load(f)

tar_data = []
for src_item in tqdm(src_data):
    valid = True

    # video_start_time = src_item["video_start_time"]
    # video_start_time = round(video_start_time)
    # if video_start_time > 0:
    #     print(video_start_time)

    conversations = src_item["conversation"]
    instruction = conversations[0]
    query_time = instruction['time']
    # query_time = float(round(query_time))      # 取整

    video_file = src_item["video_uid"]
    video_path = join('shot2story-videos', video_file)
    video_start_time = 0.0
    if video_file in video_duration_dict.keys():
        video_duration = video_duration_dict[video_file]
    else:
        video_duration = conversations[-1]['timespan'][1]

    messages, videos = [], []

    if query_time > video_start_time:
        messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, 'valid': True})
        videos.append({"file": video_path, "time": [0, query_time]})
    messages.append({"role": "user", "content": instruction["content"], 'ignore_end_stream': False, 'valid': True})
    last_time = query_time

    for idx in range(len(conversations)):
        if idx == 0:
            continue
        src_conv = conversations[idx]
        assert src_conv['role'] == 'assistant'

        src_start, src_end = src_conv['timespan']
        src_start = max(src_start, query_time)
        response = src_conv["content"]

        # 设置 positive_time, negative_time
        # positive_time 是闭区间， negative_time 是开区间
        # 默认加入一个无效的区间，防止llamafactory在数据类型上的bug
        positive_time, negative_time = [[-2.0, -1.0]], [[-2.0, -1.0]]

        # 确定回答插入的位置
        if isinstance(answer_insert_point, list) or isinstance(answer_insert_point, tuple):
            insert_point = random.uniform(answer_insert_point[0], answer_insert_point[1])
        else:
            insert_point = answer_insert_point

        response_time = src_start + insert_point * (src_end - src_start)
        if response_time <= last_time:
            print('error: 时间间隔太小，放弃')
            print(conversations)
            valid = False
            break

        video_info = {
            "file": video_path,
            "time": [last_time, response_time],
            "positive_time": [[-2.0, -1.0]],
            "negative_time": [[-2.0, -1.0]],
        }
        messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, "valid": True})
        videos.append(video_info)
        # 监督 llm loss, valid = True
        messages.append({"role": "assistant", "content": response, 'ignore_end_stream': True, "valid": True})
        last_time = response_time

    tar_item = {"messages": copy.deepcopy(messages), "videos": copy.deepcopy(videos)}
    tar_data.append(tar_item)



with open(tar_path, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)
print(len(tar_data))

# 36834