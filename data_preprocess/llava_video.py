import json
import os

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

src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/LLaVA-Video-178K/0_30_s_academic_v0_1/0_30_s_academic_oe_v0_1_qa_processed.json'
with open(src_file, 'r') as f:
    src_data = json.load(f)
print(f"{src_file}: {len(src_data)}")