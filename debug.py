import pandas as pd

df = pd.read_json("/home/SENSETIME/zengwang/myprojects/task_define_service/data/shot2story/processed/qa_gpt4o_v1.json")

print(df.dtypes)  # 查看列数据类型
print(df.head())