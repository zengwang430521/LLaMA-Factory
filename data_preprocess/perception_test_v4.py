"""
[......视频a......] Query [......视频b......] [......目标片段1......][......视频c......][......目标片段2......] [...视频d...] Answer1
------------------ -----n nnnnnnnnnnnnnnnnnn  ----------------------- YYYYYYYYYYYYYYYYY YYYYYYYYYYYYYYYYYYYYYYY YYYYYYYYYYYYY -------

[......视频a......] Query [......视频b....] [......目标片段1......] Answer1 [......视频c......][......目标片段2......][...视频d...] Answer2
----------------------------------------------------------------------------nnnnnnnnnnnnnnnnnnn ---------------------- YYYYYYYYYYYYY -------

部分数据：
Query [......目标片段1......][......目标片段2......] Answer1
-----n ---------------------- YYYYYYYYYYYYYYYYYYYYYY ----------

Query  [......目标片段1......] Answer1 [......目标片段2......][...视频d...] Answer2
-------------------------------------------------------------- YYYYYYYYYYY -------

"""
import json
from tqdm import tqdm
import random
random.seed(42)
import math
import os
import numpy as np
import copy


def get_frame_label(messages, video_duration, real_fps, mask_history=True):
    video_time_segs = []
    response_periods = []
    valids = []
    for idx, message in enumerate(messages):
        content = message["content"]
        if '<video>' in content:
            time = message['time']

            for i in range(0, len(time), 2):
                video_time_segs.append([time[i], time[i + 1]])
                response_periods.append(None)

                if mask_history and idx < len(messages) - 3:
                    valids.append(False)
                else:
                    valids.append(True)

            if idx + 1 < len(messages):
                next_role = messages[idx + 1]["role"]
                next_time = messages[idx + 1]["time"]
                if next_role == 'assistant':
                    response_periods[-1] = next_time




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
    for sample_time, response_period, valid in zip(frame_times, response_periods, valids):
        if not valid:
            frame_label = [-100] * len(sample_time)
        elif response_period is None:
            frame_label = [0] * len(sample_time)
        else:
            '''
            response_period: 表示可以进行回复的区间 [t1, t2, t3, t4]
            0:  不回复
            1:  回复
            -:  不监督
            ......t1 ...... t2 ...... t3 ......t4.......
            000000----------111111111111---------0000000
            实际上，目前起作用的只有t1, t2，因为视频不会太长
            '''

            t1, t2, t3, t4 = response_period
            frame_label = []
            for t in sample_time:
                if t < t1 or t4 < t:
                    frame_label.append(0)  # 不回复
                elif t2 <= t and t <= t3:
                    frame_label.append(1)  # 回复
                else:
                    frame_label.append(-100)  # 不监督
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


tar_file = f'/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/processed/REC_trainval.json'
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

        action_counts = {}
        for action in item["action_localisation"]:
            action_type = action['label']
            if action_type in action_counts:
                action_counts[action_type] += 1
            else:
                action_counts[action_type] = 1

        for activity in action_counts.keys():
            if activity.lower() == 'other':
                continue

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
                delta = t_end - t_start
                response_period = [t_start + 0.4 * delta, t_start + 0.6 * delta, t_end, t_end + 1]
                response_time = t_end  # 为了充分训练，插入response的位置要尽量靠后一些

                messages.append({"role": "user", "content": "<video>", "time": [last_time, response_time]})
                videos.append(video_path)
                messages.append({"role": "assistant", "content": answer, "time": response_period})
                last_time = response_time

            tar_item = {"messages": messages, "videos": videos}
            tar_data.append(tar_item)

            frame_times, frame_labels = get_frame_label(copy.deepcopy(messages), video_duration, real_fps,
                                                        mask_history=False)
            for frame_label in frame_labels:
                for l in frame_label:
                    label_count[l] += 1

print(label_count)
# os.makedirs(os.path.dirname(tar_file), exist_ok=True)
# with open(tar_file, 'w', encoding='utf-8') as f:
#     json.dump(tar_data, f, ensure_ascii=False, indent=2)
# print(len(tar_data))