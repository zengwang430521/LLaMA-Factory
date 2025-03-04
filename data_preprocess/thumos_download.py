# Thanks for submitting the access form.
# The passwords of THUMOS15 and 14 are "THUMOS15_challenge_REGISTERED" and "THUMOS14_REGISTERED", respectively.

import json
import os
from os.path import join
from collections import defaultdict
from tqdm import tqdm
import math
import random


with open('/home/SENSETIME/zengwang/myprojects/task_define_service/data/thumos/processed/filtered_videos_ids.json', 'r', encoding='utf-8') as f:
    filtered_videos_ids = json.load(f)
print(len(filtered_videos_ids))

# 下载视频
for video_id in tqdm(filtered_videos_ids):
    video_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/thumos/videos/'
    os.makedirs(video_dir, exist_ok=True)
    os.chdir(video_dir)
    if 'thumos15' in video_id:
        video_path = f'{video_dir}/{video_id}.mp4'
        video_url = f'http://storage.googleapis.com/www.thumos.info/thumos15_validation/{video_id}.mp4'
        print(video_path)
        if not os.path.exists(video_path):
            # os.system(f'wget -O {video_path} {video_url}')
            os.system(f"aria2c -x 16 -s 16 -o {video_id}.mp4 \"{video_url}\"")
            t = 0
