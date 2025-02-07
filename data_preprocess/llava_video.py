import json
import os
from os.path import join
from tqdm import tqdm

# root_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K'
# for root, dirs, files in os.walk(root_dir):
#     for file in files:
#         src_file = os.path.join(root, file)
#
#         if not src_file.endswith('.json'):
#             continue
#
#         with open(src_file, 'r') as f:
#             src_data = json.load(f)
#         print(f"{src_file}: {len(src_data)}")


tar_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/processed/v2'
os.makedirs(tar_dir, exist_ok=True)

for src_dir in [
    "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/0_30_s_academic_v0_1",
    "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/0_30_s_youtube_v0_1",
    "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/1_2_m_academic_v0_1",
    "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/1_2_m_youtube_v0_1"
]:
    for filename in os.listdir(src_dir):
        if not filename.endswith('.json'):
            continue

        src_path = join(src_dir, filename)
        tar_path = join(tar_dir, filename)

        with open(src_path, 'r') as f:
            src_data = json.load(f)

        role_transfer_dict = {
            "human": "user",
            "gpt": "assistant"
        }
        tar_data = []
        for src_item in tqdm(src_data):
            t = 0
            end_time = 100000
            video_message = {"role": "user", "content": "<video>", "time": [0, end_time]}     # 完整视频
            video_path = join('LLaVA-Video-178K', src_item['data_source'], src_item['video'])
            messages, videos = [], []
            messages.append(video_message)
            videos.append(video_path)
            for conv in src_item["conversations"]:
                role = role_transfer_dict[conv["from"]]
                content = conv["value"].replace('<image>\n', "")
                messages.append({"role": role, "content": content, "time": [end_time, end_time]})
            tar_item = {"messages": messages, "videos": videos}
            tar_data.append(tar_item)
        with open(tar_path, 'w', encoding='utf-8') as f:
            json.dump(tar_data, f, ensure_ascii=False)
        print(f"{filename}: {len(tar_data)}")




