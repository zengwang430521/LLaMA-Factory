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
tar_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/thumos/processed/REC_valid_v5_stream.json'

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

                # 设置 positive_time, negative_time
                # positive_time 是闭区间， negative_time 是开区间
                # 默认加入一个无效的区间，防止llamafactory在数据类型上的bug
                positive_time, negative_time = [[-2.0, -1.0]], [[-2.0, -1.0]]

                cur_time = last_time
                for tmp_idx in range(idx, len(actions)):
                    tmp_start, tmp_end = actions[tmp_idx]
                    tmp_start = max(tmp_start, query_time)
                    if tmp_idx < len(actions) - 1:
                        next_start, _ = actions[tmp_idx + 1]
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
                frame_times0, frame_labels0 = get_frame_label(messages, videos, video_duration, real_fps,
                                                              mask_history=True)
                frame_times, frame_labels = get_frame_label(messages, videos, video_duration, real_fps,
                                                            mask_history=False)
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
                video_info = {
                    "file": video_path,
                    "time": [last_time, response_time],
                    "positive_time": [[-2.0, -1.0]],
                    "negative_time": [[-2.0, -1.0]],
                }
                messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True, "valid": True})
                videos.append(video_info)
                # 回答插入的时机不合适，所以不监督 llm loss, valid = False
                messages.append({"role": "assistant", "content": answer, 'ignore_end_stream': True, "valid": False})
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

                frame_times0, frame_labels0 = get_frame_label(messages, videos, video_duration, real_fps,
                                                              mask_history=True)
                frame_times, frame_labels = get_frame_label(messages, videos, video_duration, real_fps,
                                                            mask_history=False)
                assert frame_labels0 == frame_labels
                for frame_label in frame_labels:
                    for l in frame_label:
                        label_count[l] += 1
                t = 0

os.makedirs(os.path.dirname(tar_file), exist_ok=True)
with open(tar_file, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)
print(len(tar_data), label_count)
# 9836 {0: 36422, 1: 23606, -100: 294735}


