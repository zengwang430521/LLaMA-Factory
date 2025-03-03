import json

src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/OVO-Bench/ovo_bench.json'
with open(src_file, 'r') as f:
    data = json.load(f)
for item in data:
    if item['task'] == "REC":
        print(item["video"])