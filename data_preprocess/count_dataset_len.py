import json
import os
from os.path import join
import shutil
import ijson
from tqdm import tqdm

# root_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/processed/v2/'
# root_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/processed/v5/'

# root_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/processed/stitch_v1/"
# root_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/processed/stitch_v2/"
# root_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/processed/stitch_v4/"

# root_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed/v2/"
# root_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed/v4/"
# root_dir = "/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed/v5/"

root_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/COIN/annotations/processed'

root_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/processed'


for filename in os.listdir(root_dir):
    if not filename.endswith('.json'):
        continue
    count = 0
    with open(join(root_dir, filename), "r") as f:
        for _, _ in tqdm(enumerate(ijson.items(f, 'item'))):
            count += 1
    if f'{count}' not in filename:
        new_name = filename.replace('.json', f'_{count}.json')
        # shutil.move(join(root_dir, filename), join(root_dir, new_name))
        print(new_name)
