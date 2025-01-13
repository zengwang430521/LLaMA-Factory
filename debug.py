from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info, fetch_video
import torch
from llamafactory.monkey_patch.qwen2_vl_monkey_patch import Qwen2VLStream
from tqdm import tqdm
import argparse
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model

model_path = '/afs/zengwang/projects/task_define_service/LLaMA-Factory/work_dirs/stream_head_only_1/Stream-Qwen2-VL-7B-Instruct'
processor = AutoProcessor.from_pretrained(model_path)

messages = [
    {"role": "system", "content": 'system_prompt'},
    {"role": "user", "content": 'query'},
    {"role": "role", "content": 'answer'},
    {"role": "user", "content": 'query'},
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
inputs = processor(text=[text], images=None, videos=None, padding=True, return_tensors="pt")
inputs = inputs.to("cuda")