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
from os.path import join, exists
import random
from tqdm import tqdm
from os.path import join
from utils import get_frame_label
import copy


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


min_pos_duration = None
answer_insert_point = 0.3
stream_positive_point = 0.7
action_extend_time = 1
action_bridge_time = 1


src_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/"
tar_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed/v5"
os.makedirs(tar_dir, exist_ok=True)
src_path = join(src_dir, 'dvc_train-human_anno-0.25_0.5_earlier.json')
tar_path = join(tar_dir, "dvc_train-human_anno-0.25_0.5_earlier_v5_stream.json")

with open(src_path, 'r') as f:
    src_data = json.load(f)

tar_data = []
label_count = {0: 0, 1: 0, -100: 0}

for video_file in tqdm(src_data.keys()):
    for _, conversations in src_data[video_file].items():
        valid = True
        video_path = join('shot2story-videos', video_file)
        instruction = random.choice(instructions)['content']
        video_duration = conversations[-1]['timespan'][1]

        messages, videos = [], []
        # 加入instruction
        query_time = 0
        if query_time > 0:
            messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, 'valid': True})
            videos.append({"file": video_path, "time": [0, query_time]})
        messages.append({"role": "user", "content": instruction, 'ignore_end_stream': False, 'valid': True})
        last_time = query_time

        for idx in range(len(conversations)):
            src_conv = conversations[idx]
            src_start, src_end = src_conv['timespan']
            src_start = max(src_start, query_time)
            response = src_conv["text"]

            # 设置 positive_time, negative_time
            # positive_time 是闭区间， negative_time 是开区间
            # 默认加入一个无效的区间，防止llamafactory在数据类型上的bug
            positive_time, negative_time = [[-2.0, -1.0]], [[-2.0, -1.0]]

            cur_time = last_time
            for tmp_idx in range(idx, len(conversations)):
                tmp_conv = conversations[tmp_idx]
                tmp_start, tmp_end = tmp_conv['timespan']
                tmp_start = max(tmp_start, query_time)

                if tmp_idx < len(conversations) - 1:
                    next_start, _ = conversations[tmp_idx + 1]['timespan']
                else:
                    next_start = video_duration

                # 相关段落之前的时间，都是负监督
                if tmp_start > cur_time:
                    negative_time.append([cur_time - 0.01, src_start])

                # 段落内，靠后部分给正监督，确定给正监督的时间点
                if isinstance(stream_positive_point, list) or isinstance(stream_positive_point, tuple):
                    pos_point = random.uniform(stream_positive_point[0], stream_positive_point[1])
                else:
                    pos_point = stream_positive_point

                # 保证段落内的正监督信号必须大于min_pos_duration
                if min_pos_duration is not None:
                    pos_point = min(pos_point, 1 - min_pos_duration / (tmp_end - tmp_start))
                    pos_point = max(pos_point, 0)

                # action 结束后一小段时间也可以回复
                extra_end = min(tmp_end + action_extend_time, next_start)
                extra_ignore = min(extra_end + action_bridge_time, next_start)

                pos_time = [tmp_start + pos_point * (tmp_end - tmp_start), extra_end]
                positive_time.append(pos_time)
                cur_time = extra_ignore

            if cur_time < video_duration:
                negative_time.append([cur_time, video_duration + 0.01])

            video_info = {
                "file": video_path,
                "time": [last_time, video_duration],
                "positive_time": positive_time,
                "negative_time": negative_time
            }
            messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, "valid": True})
            videos.append(video_info)
            messages.append({"role": "assistant", "content": '', 'ignore_end_stream': True, "valid": False})  # 占位的

            # 这里必须deepcopy，因为后续需要修改messages
            tar_item = {"messages": copy.deepcopy(messages), "videos": copy.deepcopy(videos)}
            tar_data.append(tar_item)

            # 统计一下 stream_label
            real_fps = 30.0
            frame_times0, frame_labels0 = get_frame_label(messages, videos, video_duration, real_fps, mask_history=True)
            frame_times, frame_labels = get_frame_label(messages, videos, video_duration, real_fps, mask_history=False)
            assert frame_labels == frame_labels0
            for frame_label in frame_labels:
                for l in frame_label:
                    label_count[l] += 1
            t = 0

            """
            先删除之前加入的特殊数据
            然后把这一轮的回复插在正常但靠前的位置，便于后一轮的监督
            同时让这一轮的 stream label 都是 无监督-
            并且query最后的stream label只需要监督一次就好，避免多次监督造成bias
            """
            messages = messages[:-2]
            videos = videos[:-1]
            for msg in messages:
                msg['ignore_end_stream'] = True

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
            # 回答插入的时机不合适，所以不监督 llm loss, valid = False
            messages.append({"role": "assistant", "content": response, 'ignore_end_stream': True, "valid": False})
            last_time = response_time


        if last_time < video_duration:
            # 用来训练全部完成回复之后要保持沉默
            # fake answer 不能用来训练lm_head
            video_info = {
                "file": video_path,
                "time": [last_time, video_duration],
                "positive_time": [[-2.0, -1.0]],
                "negative_time": [[last_time - 0.01, video_duration + 0.01]]
            }
            messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, "valid": True})
            videos.append(video_info)
            messages.append({"role": "assistant", "content": '', 'ignore_end_stream': True, "valid": False})

            tar_item = {"messages": copy.deepcopy(messages), "videos": copy.deepcopy(videos)}
            tar_data.append(tar_item)

            frame_times0, frame_labels0 = get_frame_label(messages, videos, video_duration, real_fps, mask_history=True)
            frame_times, frame_labels = get_frame_label(messages, videos, video_duration, real_fps, mask_history=False)
            assert frame_labels0 == frame_labels
            for frame_label in frame_labels:
                for l in frame_label:
                    label_count[l] += 1
            t = 0

with open(tar_path, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)
print(label_count, len(tar_data))

# {0: 349766, 1: 568658, -100: 2276886} 193619
