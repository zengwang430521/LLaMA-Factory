import os
import json
from call_llm import call_llm_completion, call_chatapi_model, call_openai_model, call_silionflow_model, call_vllm_model
from os.path import join, exists
from concurrent.futures import ProcessPoolExecutor
import concurrent
from tqdm import tqdm
import re
import random
import math


prompt_template = """
There is a video, its content is:
{whole_caption}

This video consists of several clips. The content of these clips is: {clip_captions}

Please propose a question that ordinary audiences would ask. \
This question must be related to only a part of the clips and \
must not be related to every clips. You question must can be \
answered from the captions.
Please also give the related clips and answers. \
All your answers constitute a complete answer. \
So do not repeat the contents appeared in previous answers. \
Only include new information in the related clip.
The question and answer should not be related to the voice.

Please reply in the following json format:
{{
  "question": "the question you ask",
  "related_clips": list of related clip index
  "answers": list of the answers for related clips. Do not mention clip in the answer.
}}
"""


def generate_qa(item, save_dir):
    video = item["video"]
    output_file = join(save_dir, f'{video}.json')
    if exists(output_file):
        print(f'Exists: {output_file}')
        return

    clip_captions = ""
    for i, caption in enumerate(item["captions"]):
        clip_captions = f"{clip_captions}\nClip {i + 1}: {caption}"
    prompt = prompt_template.format(
        whole_caption=item['whole_caption'],
        clip_captions=clip_captions
    )
    messages = [{"role": "user", "content": prompt}]

    # response = call_llm_completion(messages=messages, model="SenseChat-Vision-102b-stage")
    # response = call_chatapi_model(messages=messages, model="DeepSeek-R1-Enterprise")
    # response = call_silionflow_model(messages=messages, model="deepseek-ai/DeepSeek-R1")
    # print(response)

    # result = call_openai_model(messages=messages)
    # response, cost = result["response"], result["cost"]

    print(prompt)
    response = call_vllm_model(messages=messages, model='/afs/zengwang/ckpt/DeepSeek-R1-Distill-Llama-70B')
    result = {"response": response, "cost": 0}
    print(response)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def generate_qa_tmp(item, save_dir):
    try:
        generate_qa(item, save_dir)
    except Exception as e:
        print(f"Error processing item {item}: {e}")
    return


def get_clip_frame(clip_name):
    matches = re.findall(r'_(\d+)', clip_name)
    if matches:
        numbers = list(map(int, matches))
    return numbers[-2:]


if __name__ == '__main__':
    with open('/afs/zengwang/projects/task_define_service/data/shot2story/134k_full_train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[10000:]     #跳过gpt4o的部分

    target_nums = 10000
    data = data[:target_nums]

    save_dir = '/afs/zengwang/projects/task_define_service/data/shot2story/qas_ds_70b'
    os.makedirs(save_dir, exist_ok=True)

    # for item in data:
    #     generate_qa_tmp(item, save_dir)

    max_workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_qa_tmp, item, save_dir) for item in data]
        concurrent.futures.wait(futures)

    # # 计算消耗
    # num_all, cost_all = 0, 0
    # for result_file in tqdm(os.listdir(save_dir)):
    #     if not result_file.endswith('.json'):
    #         continue
    #     with open(join(save_dir, result_file), 'r', encoding='utf-8') as f:
    #         result = json.load(f)
    #     cost_all += result['cost']
    #     num_all += 1
    # print(f'Cost: {cost_all} $ / {num_all} data = {cost_all/num_all} $/data')



    # '''汇总QA数据'''
    # video_info_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/shot2story_video_info.json'
    # with open(video_info_file, 'r') as f:
    #     video_infos = json.load(f)
    #
    # # 处理数据
    # tar_data = []
    # tar_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/shot2story/processed/qa_gpt4o_v1.json'
    # for item in tqdm(data):
    #     video = item['video']
    #     result_file = join(save_dir, f"{video}.json")
    #     if not exists(result_file):
    #         continue
    #
    #     with open(result_file, 'r', encoding='utf-8') as f:
    #         result = json.load(f)
    #     response = result['response']
    #     if response is None:
    #         continue
    #
    #     valid = True
    #     if response.startswith('```json'):
    #         response = response[len("```json"):-len("```")]
    #     result = json.loads(response)
    #
    #     video_info = video_infos[join('data', 'shot2story-videos', video)]
    #     clip_names = item["video_names"]
    #     clip_frames = [get_clip_frame(clip) for clip in clip_names]
    #     fps = video_info['frame_rate']
    #     clip_times = [[st/fps, ed/fps] for st, ed in clip_frames]
    #
    #     question = result["question"]
    #     related_clips = result["related_clips"]
    #     answers = result["answers"]
    #
    #     related_clips, answers = zip(*sorted(zip(related_clips, answers)))
    #     related_clips = list(related_clips)
    #     answers = list(answers)
    #     related_clips = [tmp-1 for tmp in related_clips]
    #
    #     # 在第一个相关片段前提问
    #     video_path = join('shot2story-videos', video)
    #     first_response_time = clip_times[related_clips[0]][1]
    #     query_time = float(random.randint(0, int(math.floor(first_response_time- 1))))
    #     messages, videos = [], []
    #     if query_time > 0:
    #         messages.append({"role": "user", "content": "<video>", "time": [0.0, query_time]})
    #         videos.append(video_path)
    #     messages.append({"role": "user", "content": question, "time": [query_time, query_time]})
    #
    #     last_time = query_time
    #     for related_clip, answer in zip(related_clips, answers):
    #         if not isinstance(answer, str):
    #             print(answer)
    #             valid = False
    #             continue
    #         response_time = float(clip_times[related_clip][1])
    #         messages.append({"role": "user", "content": "<video>", "time": [last_time, response_time]})
    #         videos.append(video_path)
    #         messages.append({"role": "assistant", "content": answer, "time": [response_time, response_time]})
    #         last_time = response_time
    #
    #     # # 剩余的视频如果大于1s，就加入，对于训练stream_head有帮助
    #     # # 有bug
    #     # if last_time < video_info['duration'] - 1:
    #     #     messages.append({"role": "user", "content": "<video>", "time": [last_time,  float(video_info['duration'])]})
    #     #     videos.append(video_path)
    #
    #     if valid:
    #         tar_item = {"messages": messages, "videos": videos}
    #         tar_data.append(tar_item)
    #
    # os.makedirs(os.path.dirname(tar_file), exist_ok=True)
    # with open(tar_file, 'w', encoding='utf-8') as f:
    #     json.dump(tar_data, f, ensure_ascii=False)
    # print(len(tar_data))
