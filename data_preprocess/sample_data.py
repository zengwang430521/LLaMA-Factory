import os
import json
from os.path import join
import numpy as np
from tqdm import tqdm
import ijson


# src_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/processed/v2/'
# sample_items = [
#     ("0_30_s_academic_mc_v0_1_qa_processed_5753.json",  5000),
#     ("0_30_s_academic_oe_v0_1_qa_processed_48468.json", 10000),
#     ("0_30_s_academic_v0_1_cap_processed_11985.json",   10000),
#     ("0_30_s_youtube_mc_v0_1_qa_processed_39353.json",  5000),
#     ("0_30_s_youtube_oe_v0_1_qa_processed_420200.json", 10000),
#     ("0_30_s_youtube_v0_1_cap_processed_79346.json",    10000),
# ]


# src_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/processed/v2/'
# sample_items = [
#     ("0_30_s_academic_mc_v0_1_qa_processed_5753.json",  2000),
#     ("0_30_s_academic_oe_v0_1_qa_processed_48468.json", 4000),
#     ("0_30_s_academic_v0_1_cap_processed_11985.json",   4000),
#     ("0_30_s_youtube_mc_v0_1_qa_processed_39353.json",  2000),
#     ("0_30_s_youtube_oe_v0_1_qa_processed_420200.json", 4000),
#     ("0_30_s_youtube_v0_1_cap_processed_79346.json",    4000),
# ]


# src_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed/v2/'
# sample_items = [
#     ("dvc_train-human_anno-0.25_0.5_earlier_36948.json",    25000),
#     ("magqa_train-0.25_0.5-earlier_36834.json",             25000),
# ]


src_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed/v4/'
sample_items = [
    ("dvc_train-human_anno-0.25_0.5_earlier_36949.json",    10000),
    ("magqa_train-0.25_0.5-earlier_36834.json",             10000),
]



tar_dir = join(src_dir, 'sample')
os.makedirs(tar_dir, exist_ok=True)
for item in sample_items:
    src_file, tar_num = item
    print(src_file)
    src_num = int(src_file.split('.json')[0].split('_')[-1])
    sample_idx = np.linspace(0, src_num-1, tar_num).round().astype(np.int32)
    tar_data = []

    with open(join(src_dir, src_file), 'r') as f:
        for i, d in tqdm(enumerate(ijson.items(f, 'item', use_float=True))):
            if i in sample_idx:
                tar_data.append(d)
    tar_file = '_'.join(src_file.split('_')[:-1] + [f'{len(tar_data)}.json'])
    with open(join(tar_dir, tar_file), 'w', encoding='utf-8') as f:
        json.dump(tar_data, f, ensure_ascii=False)
