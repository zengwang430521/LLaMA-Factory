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
from tqdm import tqdm
from os.path import join
import random
import ijson
import math
from collections import defaultdict


stream_positive_point = 0.7

# tar_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/processed/stitch_v1'
tar_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/processed/stitch_v5'
os.makedirs(tar_dir, exist_ok=True)

# question_type = 'oe'
question_type = 'mc'
short_threshold = 8
target_num = 20000

tar_file = join(tar_dir, f'0_30s_{question_type}.json')

video_info_file = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/llava_video_info.json"
with open(video_info_file, 'r', encoding='utf-8') as f:
    video_info_dict = json.load(f)

short_videos = []
video_splits = defaultdict(list)

for video in video_info_dict.keys():
    if not (video.endswith('.mp4') or video.endswith('.avi') or video.endswith('.mkv')):
        continue

    if "error" in video_info_dict[video].keys():
        continue

    duration = video_info_dict[video]["duration"]
    resolution = video_info_dict[video]["resolution"]
    resolution = tuple(resolution)
    if duration < short_threshold:
        short_videos.append(video)
    video_splits[resolution].append(video)

# 获取问题
short_videos = set(short_videos)
short_src_data = []
for src_dir in [
    "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/0_30_s_youtube_v0_1",
    "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/0_30_s_academic_v0_1",
]:
    for filename in os.listdir(src_dir):
        if filename.endswith('.json') and f'_{question_type}_' in filename:
            src_path = join(src_dir, filename)
            with open(src_path, 'r') as f:
                for src_item in tqdm(ijson.items(f, 'item')):
                    video_path = join('data', 'LLaVA-Video-178K', src_item['data_source'], src_item['video'])
                    if video_path in short_videos:
                        short_src_data.append(src_item)


role_transfer_dict = {
    "human": "user",
    "gpt": "assistant"
}


end_src_data = random.sample(short_src_data, min(target_num, len(short_src_data)))
qa_per_src = math.ceil(1.0 * target_num / len(end_src_data))
# end_time = 100000  #用一个比较大的值表示视频末尾

tar_data = []
for src_item in tqdm(end_src_data):
    end_video = join('data', 'LLaVA-Video-178K', src_item['data_source'], src_item['video'])
    end_video_duration = video_info_dict[end_video]['duration']
    end_video_path = end_video.split('data/')[1]

    # 前面拼接resolution相同的视频，不能拼接自己
    flag_video_merge = True
    end_video_resolution = tuple(video_info_dict[end_video]['resolution'])
    start_videos = set(video_splits[end_video_resolution])
    start_videos.remove(end_video)

    # 如果这个尺寸的视频只有自己，就只能找其他视频，而且不可以进行视频拼接
    if len(start_videos) == 0:
        flag_video_merge = False
        print(end_video_resolution)
        start_videos = []
        for resolution in video_splits.keys():
            if resolution != end_video_resolution:
                start_videos += video_splits[resolution]

    qas = []
    q, a = None, None
    for conv in src_item["conversations"]:
        role = role_transfer_dict[conv["from"]]
        content = conv["value"].replace('<image>\n', "")
        if role == 'user':
            q = content
        elif role == 'assistant':
            a = content
            if a != 'None':
                qas.append((q, a))
    sampled_qas = random.sample(qas, min(len(qas), qa_per_src))
    for q, a in sampled_qas:
        start_video = random.sample(start_videos, 1)[0]
        start_video_path = start_video.split('data/')[1]
        start_video_duration = video_info_dict[start_video]["duration"]

        query_time = random.randint(2, int(start_video_duration-2))

        # 段落内，靠后部分给正监督，确定给正监督的时间点
        if isinstance(stream_positive_point, list) or isinstance(stream_positive_point, tuple):
            pos_point = random.uniform(stream_positive_point[0], stream_positive_point[1])
        else:
            pos_point = stream_positive_point

        videos = [
            # start video 前半段
            {
                "file": start_video_path,
                "time": [0, query_time],
                "positive_time": [[-2.0, -1.0]],
                "negative_time": [[-2.0, -1.0]]
            },
            # start video 后半段
            {
                "file": start_video_path,
                "time": [query_time, start_video_duration],
                "positive_time": [[-2.0, -1.0]],
                "negative_time": [[query_time-0.01, start_video_duration+0.01]]
            },
            # end video
            {
                "file": end_video_path,
                "time": [0, end_video_duration],
                "positive_time": [[pos_point*end_video_duration, end_video_duration]],
                "negative_time": [[-2.0, -1.0]]
            },
        ]

        if flag_video_merge:
            messages = [
                {"role": "user", "content": "<video>", "ignore_end_stream": True, "valid": True},           # start video 前半段
                {"role": "user", "content": q, "ignore_end_stream": False, "valid": True},                  # query
                {"role": "user", "content": "<video><+><video>", "ignore_end_stream": True, "valid": True}, # start video 后半段 + end video,可以拼接
                {"role": "assistant", "content": a, "ignore_end_stream": True, "valid": True},
            ]
        else:
            messages = [
                {"role": "user", "content": "<video>", "ignore_end_stream": True, "valid": True},   # start video 前半段
                {"role": "user", "content": q, "ignore_end_stream": False, "valid": True},          # query
                {"role": "user", "content": "<video>", "ignore_end_stream": True, "valid": True},   # start video 后半段
                {"role": "user", "content": "<video>", "ignore_end_stream": True, "valid": True},   # end video,不可以 start video 拼接
                {"role": "assistant", "content": a, "ignore_end_stream": True, "valid": True},      # 回答
            ]

        tar_item = {"messages": messages, "videos": videos}
        tar_data.append(tar_item)

with open(tar_file, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False)
print(len(tar_data))
t = 0