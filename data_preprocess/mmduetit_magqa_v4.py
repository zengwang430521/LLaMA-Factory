import json
import os
from os.path import join, exists
import random
from tqdm import tqdm

src_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/"
tar_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed/v4"
os.makedirs(tar_dir, exist_ok=True)
src_path = join(src_dir, 'magqa_train-0.25_0.5-earlier.json')
tar_path = join(tar_dir, "magqa_train-0.25_0.5-earlier.json")

with open(src_path, 'r') as f:
    src_data = json.load(f)

tar_data = []
for src_item in tqdm(src_data):
    valid = True

    # video_start_time = src_item["video_start_time"]
    # video_start_time = round(video_start_time)
    # if video_start_time > 0:
    #     print(video_start_time)
    video_start_time = 0

    messages, videos = [], []

    video_file = src_item["video_uid"]
    video_path = join('shot2story-videos', video_file)

    conversations = src_item["conversation"]
    instruction = conversations[0]
    query_time = instruction['time']
    query_time = round(query_time)      # 取整
    if query_time > video_start_time:
        video_message = {"role": "user", "content": "<video>", "time": [video_start_time, query_time]}
        messages.append(video_message)
        videos.append(video_path)

    query_message = {"role": "user", "content": instruction["content"], "time": [query_time, query_time]}
    messages.append(query_message)
    last_time = query_time

    for idx_resp, src_conv in enumerate(conversations[1:]):
        # response_time = round(src_conv['timespan'][1])
        # response_time = round(response_time)  # 取整
        # response_time = response_time

        '''
        response_period: 表示可以进行回复的区间 [t1, t2, t3, t4]
        0:  不回复
        1:  回复
        -:  不监督
        ......t1 ...... t2 ...... t3 ......t4.......
        000000----------111111111111---------0000000
        '''
        t_start, t_end = src_conv['timespan']
        delta = t_end - t_start
        response_period = [t_start + 0.4 * delta, t_start + 0.6 * delta, t_end, t_end + 1 ]
        response_time = t_end               # 为了充分训练，插入response的位置要尽量靠后一些

        if response_time > last_time:
            video_message = {"role": "user", "content": "<video>", "time": [last_time, response_time]}
            messages.append(video_message)
            videos.append(video_path)
            response_message = {"role": "assistant", "content": src_conv["content"], "time": response_period}
            messages.append(response_message)
            last_time = response_time
        else:
            if idx_resp == 0:
                # print('当场回答')
                response_message = {"role": "assistant", "content": src_conv["content"], "time": [response_time, response_time]}
                messages.append(response_message)
                last_time = response_time
            else:
                print('error: 两次回答之间时间间隔太小，放弃')
                valid = False
                break
    if valid:
        tar_item = {"messages": messages, "videos": videos}
        tar_data.append(tar_item)

with open(tar_path, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False)
print(len(tar_data))
