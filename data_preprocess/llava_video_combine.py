import json
import os
from tqdm import tqdm
from os.path import join
import random
import ijson
import math


tar_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/processed/stitch_v1'
os.makedirs(tar_dir, exist_ok=True)

question_type = 'oe'
# question_type = 'mc'
short_threshold = 8
target_num = 20000

tar_file = join(tar_dir, f'0_30s_{question_type}.json')

# 只保留同一个子集的视频
duration_file = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/llava_video_len.json"
with open(duration_file, 'r', encoding='utf-8') as f:
    duartion_dict = json.load(f)

short_videos = set([])
for video in duartion_dict:
    if duartion_dict[video] < short_threshold:
        short_videos.add(video)


# 获取问题
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
start_videos = random.sample(list(duartion_dict.keys()), len(end_src_data))
end_time = 100000  #用一个比较大的值表示视频末尾

tar_data = []
for start_video, src_item in tqdm(zip(start_videos, end_src_data)):
    end_video = join('data', 'LLaVA-Video-178K', src_item['data_source'], src_item['video'])
    start_video_duration = duartion_dict[start_video]
    end_video_duration = duartion_dict[end_video]

    start_video_path = start_video.split('data/')[1]
    end_video_path = end_video.split('data/')[1]

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
        query_time = random.randint(2, int(start_video_duration-2))
        messages, videos = [], []
        messages = [
            {"role": "user", "content": "<video>", "time": [0, query_time]},                                # start video 前半段
            {"role": "user", "content": q, "time": [query_time, query_time]},                     # query
            {"role": "user", "content": "<video>", "time": [query_time, start_video_duration]},   # start video 后半段
            {"role": "user", "content": "<video>", "time": [0, end_video_duration]},                        # end video
            {"role": "assistant", "content": a, "time": [end_video_duration, end_video_duration]},          # 回答
        ]
        videos = [
            start_video_path,
            start_video_path,
            end_video_path
        ]
        tar_item = {"messages": messages, "videos": videos}
        tar_data.append(tar_item)

with open(tar_file, 'w', encoding='utf-8') as f:
    json.dump(tar_data, f, ensure_ascii=False)
print(len(tar_data))
t = 0