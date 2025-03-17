import json


def remove_first_last_testinfo(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "test_info" in item and len(item["test_info"]) > 2:
            item["test_info"] = item["test_info"][1:-1]  # 删除第一个和最后一个元素
        elif "test_info" in item:
            item["test_info"] = []  # 若不足3个元素，则清空

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data

remove_first_last_testinfo("/home/SENSETIME/zengwang/myprojects/task_define_service/data/perception_test/processed/REC_trainval_ovo_test.json")