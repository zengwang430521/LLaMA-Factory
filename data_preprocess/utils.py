import copy
import math
import numpy as np


def get_frame_label(messages, videos, video_duration, real_fps, mask_history=True):
    messages = copy.deepcopy(messages)
    videos = copy.deepcopy(videos)
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