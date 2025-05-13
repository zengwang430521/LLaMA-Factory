import json
import os
from os.path import join
from tqdm import tqdm
import ijson


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


tar_dir = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/processed/v5'
os.makedirs(tar_dir, exist_ok=True)

for src_dir in [
    "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/0_30_s_academic_v0_1",
    "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/0_30_s_youtube_v0_1",
    # "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/1_2_m_academic_v0_1",
    # "/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/1_2_m_youtube_v0_1"
]:
    for filename in os.listdir(src_dir):
        if not filename.endswith('.json'):
            continue

        src_path = join(src_dir, filename)
        tar_path = join(tar_dir, filename)

        role_transfer_dict = {
            "human": "user",
            "gpt": "assistant"
        }
        tar_data = []

        err_num = 0
        with open(src_path, 'r') as f:
            for src_item in tqdm(ijson.items(f, 'item')):
                end_time = 100000.0
                video_path = join('LLaVA-Video-178K', src_item['data_source'], src_item['video'])
                video_info = {
                    "file": video_path,
                    "time": [0.0, end_time],
                    "positive_time": [[-2.0, -1.0]],
                    "negative_time": [[-2.0, -1.0]],
                }

                messages, videos = [], []
                messages.append({"role": "user", "content": "<video>", "ignore_end_stream": True, "valid": True} )
                videos.append(video_info)

                for conv in src_item["conversations"]:
                    role = role_transfer_dict[conv["from"]]
                    content = conv["value"].replace('<image>\n', "")

                    if role == 'user':
                        q = content
                    elif role == 'assistant':
                        a = content
                        if a != 'None':
                            messages.append({"role": 'user', "content": q, "ignore_end_stream": False, "valid": True})
                            messages.append({"role": 'assistant', "content": a, "ignore_end_stream": False, "valid": True})
                        else:
                            err_num += 1

                tar_item = {"messages": messages, "videos": videos}
                tar_data.append(tar_item)

        with open(tar_path, 'w', encoding='utf-8') as f:
            json.dump(tar_data, f, ensure_ascii=False)
        print(f"{filename}: {len(tar_data)}")
        print(err_num)



