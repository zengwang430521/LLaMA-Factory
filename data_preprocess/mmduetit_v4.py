import json
import os
from os.path import join, exists
import random
from tqdm import tqdm
from os.path import join


instructions = [
    {"role": "user", "content": "Please concisely narrate the video in real time."},
    {"role": "user", "content": "Help me to illustrate my view in short."},
    {"role": "user", "content": "Please simply describe what do you see."},
    {"role": "user", "content": "Continuously answer what you observed with simple text."},
    {"role": "user", "content": "Do concise real-time narration."},
    {"role": "user", "content": "Hey assistant, do you know the current video content? Reply me concisely."},
    {"role": "user", "content": "Simply interpret the scene for me."},
    {"role": "user", "content": "What can you tell me about? Be concise."},
    {"role": "user", "content": "Use simple text to explain what is shown in front of me."},
    {"role": "user", "content": "What is the action now? Please response in short."},
]

src_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/"
tar_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed/v4"
os.makedirs(tar_dir, exist_ok=True)
src_path = join(src_dir, 'dvc_train-human_anno-0.25_0.5_earlier.json')
tar_path = join(tar_dir, "dvc_train-human_anno-0.25_0.5_earlier.json")

with open(src_path, 'r') as f:
    src_data = json.load(f)

tar_data = []
for video_file in tqdm(src_data.keys()):

    for _, conversations in src_data[video_file].items():
        video_path = join('shot2story-videos', video_file)

        valid = True
        instruction = random.choice(instructions)
        messages, videos = [], []

        messages.append({"role": "user", "content": instruction["content"], "time": [0, 0]})
        last_time = 0
        for src_conv in conversations:
            # response_time = src_conv['timespan'][1]
            # response_time = round(response_time)    # 取整

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
            response_period = [t_start + 0.4 * delta, t_start + 0.6 * delta, t_end, t_end + 1]
            response_period = [float(t) for t in response_period]
            response_time = t_end  # 为了充分训练，插入response的位置要尽量靠后一些

            if response_time <= last_time:
                print('error: 时间间隔太小，放弃')
                valid = False
                break

            video_message = {"role": "user", "content": "<video>", "time": [last_time, response_time]}
            messages.append(video_message)
            videos.append(video_path)
            response_message = {"role": "assistant", "content": src_conv["text"], "time": response_period}
            messages.append(response_message)
            last_time = response_time

        if valid:
            tar_item = {"messages": messages, "videos": videos}
            tar_data.append(tar_item)

with open(tar_path, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)
print(len(tar_data))
