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

import copy
import json

import numpy as np
from tqdm import tqdm
import random
import math
import os


def get_frame_label(messages, video_duration, real_fps, mask_history=True):
    video_files = [v['file'] for v in videos]
    video_time_segs = [v['time'] for v in videos]
    valids = []

    for idx, message in enumerate(messages):
        content = copy.deepcopy(message["content"])
        while '<video>' in content:
            if mask_history and idx < len(messages) - 3:
                valids.append(False)
            else:
                valids.append(True)
            content = content.replace("<video>", "", 1)

    # 先处理一下time_seg
    total_duration = 0
    for i in range(len(video_time_segs)):
        t_start, t_end = video_time_segs[i]
        t_start, t_end = max(t_start, 0), min(t_end, video_duration)
        total_duration += t_end - t_start
        video_time_segs[i] = [t_start, t_end]

    video_fps = 2
    video_maxlen = 64

    frame_nums = []
    for video, time_seg in zip(videos, video_time_segs):
        # 先计算这一段需要采样多少帧
        t_start, t_end = time_seg
        seg_duration = t_end - t_start
        frame_num = min(seg_duration * video_fps, seg_duration * real_fps)
        frame_num = min(frame_num, video_maxlen * seg_duration / total_duration)
        frame_num = math.floor(frame_num)
        frame_num = max(frame_num, 2)  # 最少采集2帧
        if frame_num % 2 != 0:
            # 必须是偶数
            frame_num -= 1
        frame_nums.append(frame_num)

    # 此时各段的采样帧数可能加起来超过 video_maxlen
    current_total = sum(frame_nums)
    # 如果超过，则对各段进行迭代调整，每次从那些帧数大于2的段减少2帧，直到总数不超过总数要求
    while current_total > video_maxlen:
        # import pdb; pdb.set_trace()
        # print('DEBUG: frame index!')

        reduced = False
        for i in range(len(frame_nums)):
            if frame_nums[i] > 2:
                frame_nums[i] -= 2
                current_total -= 2
                reduced = True
                if current_total <= video_maxlen:
                    break
        if not reduced:
            # 如果所有段都已经是2帧，无法再减少，则退出循环
            break

    # 确定采样的frame idx
    frame_times =  []
    for video, time_seg, frame_num in zip(videos, video_time_segs, frame_nums):
        t_start, t_end = time_seg
        sample_times = np.linspace(t_start, t_end, frame_num + 1)[1:]
        frame_times.append(sample_times[1::2])

    # 判断哪些帧需要回答
    frame_labels = []
    for sample_time, video, valid in zip(frame_times, videos, valids):
        frame_label = [-100] * len(sample_time)
        if valid:
            # positive 用闭区间
            positive_time = video.get('positive_time', None)
            if positive_time is not None:
                for t_start, t_end in positive_time:
                    for i, t in enumerate(sample_time):
                        if t_start <= t <= t_end:
                            frame_label[i] = 1

            # negative 用开区间
            negative_time = video.get('negative_time', None)
            if negative_time is not None:
                for t_start, t_end in negative_time:
                    for i, t in enumerate(sample_time):
                        if t_start < t < t_end:
                            frame_label[i] = 0
        frame_labels.append(frame_label)

    return frame_times, frame_labels


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



# ignore_single_action = False
# answer_insert_point = 0.3
# stream_positive_point = 0.7
# action_extend_time = 2
# action_bridge_time = 2
# tar_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/processed/REC_trainval_stream_only_v5_2.json'


ignore_single_action = False
answer_insert_point = [0.1, 0.5]
stream_positive_point = 0.6
action_extend_time = 1
action_bridge_time = 1
tar_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/processed/REC_trainval_stream_only_v5_4.json'

tar_data = []
label_count = {0: 0, 1: 0, -100: 0}

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
        video_duration = (item['metadata']['num_frames'] - 1) / item['metadata']['frame_rate']
        real_fps = item['metadata']['frame_rate']
        frame_interval = 1.0 / real_fps

        action_counts = {}
        for action in item["action_localisation"]:
            action_type = action['label']
            if action_type in action_counts:
                action_counts[action_type] += 1
            else:
                action_counts[action_type] = 1

        for activity in action_counts.keys():
            # 只利用出现了多次的动作生成计数数据
            if ignore_single_action and action_counts[activity] < 2:
                continue

            filtered_action = [action for action in item["action_localisation"] if action['label'] == activity]
            filtered_action_times = [[action['timestamps'][0] * 1e-6, action['timestamps'][1] * 1e-6] for action in filtered_action]

            query = query_template.format(activity=activity.lower())
            first_response_time = filtered_action_times[0][0]

            max_query_time = math.floor(first_response_time - 3)
            max_query_time = max(max_query_time, 0)
            query_time = float(random.randint(0, int(max_query_time)))

            messages, videos = [], []
            if query_time > 0:
                messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True})
                videos.append({"file": video_path, "time": [0, query_time]})
            messages.append({"role": "user", "content": query})

            # DEBUG 用
            if len(filtered_action_times) >= 3:
                t = 0

            last_time = query_time
            for idx in range(len(filtered_action_times)):
                count = idx + 1
                answer = str(count)

                # 遍历之后的时间和action, 设置 positive_time, negative_time
                # positive_time 是闭区间， negative_time 是开区间
                positive_time, negative_time = [], []

                # negative_time 是开区间，所以设置的时候略微提前一点, 确保覆盖第一帧
                cur_time = last_time - frame_interval

                for tmp_idx in range(idx, len(filtered_action_times)):
                    tmp_start, tmp_end = filtered_action_times[tmp_idx]
                    if tmp_idx < len(filtered_action_times) - 1:
                        next_start, _ = filtered_action_times[tmp_idx + 1]
                    else:
                        next_start = video_duration

                    # 无关视频不回复, 因为是闭区间，所以不要包含 tmp_start
                    if cur_time < tmp_start:
                        negative_time.append([cur_time, tmp_start])

                    # action 靠前部分不监督

                    # action 靠后部分可以回复
                    if isinstance(stream_positive_point, list) or isinstance(stream_positive_point, tuple):
                        pos_point = random.uniform(stream_positive_point[0], stream_positive_point[1])
                    else:
                        pos_point = stream_positive_point
                    positive_time.append([tmp_start + pos_point * (tmp_end-tmp_start), tmp_end])

                    # action 结束后一小段时间可以回复, 但是不要超过下一段action
                    extra_end = min(tmp_end + action_extend_time, next_start)
                    if extra_end > tmp_end:
                        positive_time.append([tmp_end, extra_end])

                    # extra_end 之后一段时间不要监督
                    extra_ignore = min(extra_end + action_bridge_time, next_start)

                    cur_time = extra_ignore

                # 没有action了，直到视频结束都不回复
                # negative_time 是开区间，所以设置的时候略微靠后一点，确保覆盖最后1帧
                if cur_time < video_duration:
                    negative_time.append([cur_time, video_duration + frame_interval])

                act_start, act_end = filtered_action_times[idx]

                video_info = {
                    "file": video_path,
                    "time": [last_time, video_duration],
                    "positive_time": positive_time,
                    "negative_time": negative_time
                }
                messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True})
                videos.append(video_info)
                messages.append({"role": "assistant", "content": answer})

                # 这里必须deepcopy，因为后续需要修改messages
                tar_item = {"messages": copy.deepcopy(messages), "videos": copy.deepcopy(videos)}
                tar_data.append(tar_item)

                # 统计一下 stream_label
                frame_times, frame_labels = get_frame_label(copy.deepcopy(messages), video_duration, real_fps, mask_history=True)
                for frame_label in frame_labels:
                    for l in frame_label:
                        label_count[l] += 1
                t = 0

                """
                先删除之前加入的特殊数据
                然后把这一轮的回复插在正常但靠前的位置（前25%的位置)，便于后一轮的监督
                同时让这一轮的 stream label 都是 无监督-
                """
                messages = messages[:-2]
                videos = videos[:-1]

                if isinstance(answer_insert_point, list) or isinstance(answer_insert_point, tuple):
                    insert_point = random.uniform(answer_insert_point[0], answer_insert_point[1])
                else:
                    insert_point = answer_insert_point

                response_time = act_start + insert_point * (act_end - act_start)

                video_info = {
                    "file": video_path,
                    "time": [last_time, response_time],
                }
                messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True})
                videos.append(video_info)
                messages.append({"role": "assistant", "content": answer})

                last_time = response_time

            if last_time < video_duration:
                # 用来训练全部完成回复之后要保持沉默
                # fake answer 不能用来训练lm_head

                video_info = {
                    "file": video_path,
                    "time": [last_time, video_duration],
                    "positive_time": [],
                    "negative_time": [[last_time-frame_interval, video_duration+frame_interval]]
                }
                messages.append({"role": "user", "content": "<video>", 'ignore_end_stream': True})
                videos.append(video_info)
                messages.append({"role": "assistant", "content": '', "valid": False})

                tar_item = {"messages": copy.deepcopy(messages), "videos": copy.deepcopy(videos)}
                tar_data.append(tar_item)
                frame_times, frame_labels = get_frame_label(copy.deepcopy(messages), video_duration, real_fps, mask_history=True)
                for frame_label in frame_labels:
                    for l in frame_label:
                        label_count[l] += 1
                t = 0

print(label_count)
print(len(tar_data))

os.makedirs(os.path.dirname(tar_file), exist_ok=True)
with open(tar_file, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False, indent=2)


# {0: 743316, 1: 178116, -100: 780671} 73091