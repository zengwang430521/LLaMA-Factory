# 调用llm和vllm
import jwt
import time
import requests
import json
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from qwen_vl_utils import process_vision_info


from qwen_vl_utils.vision_process import *
def _read_video_decord_v2(
    ele: dict,
) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    # import pdb; pdb.set_trace()
    import decord
    video_path = ele["video"]
    start_time, end_time = ele.get("video_start", None), ele.get("video_end", None)
    st = time.time()
    vr = decord.VideoReader(video_path)

    # TODO: support start_pts and end_pts
    # if 'video_start' in ele or 'video_end' in ele:
    #     raise NotImplementedError("not support start_pts and end_pts in decord for now.")

    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    idx_start, idx_end = 0, total_frames - 1
    if start_time is not None:
        idx_start = max(round(start_time * video_fps), idx_start)
    if end_time is not None:
        idx_end = min(round(end_time * video_fps), idx_end)

    nframes = smart_nframes(ele, total_frames=(idx_end - idx_start + 1), video_fps=video_fps)
    idx = torch.linspace(idx_start, idx_end, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


import qwen_vl_utils
qwen_vl_utils.vision_process.VIDEO_READER_BACKENDS['decord'] = _read_video_decord_v2


def encode_jwt_token(ak, sk):
    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800,  # 填写您期望的有效时间，此处示例代表当前时间+30分钟
        "nbf": int(time.time()) - 120,  # 填写您期望的生效时间，此处示例代表当前时间-120秒
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token


def call_sensenova_model(messages, model, **kwargs):
    """
    调用通过sensenova上部署的模型

    Args:
        messages

    Returns:
        resp (str): 模型回复
    """
    sensenova_url = 'https://api.sensenova.cn/v1/llm/chat-completions'
    ak, sk = "196FFD00B6184227B65B3D92C01A8724", "DD1D004D80834448B276F125F8310F2A"

    api_secret_key = encode_jwt_token(ak, sk)
    data = {
        "model": model,
        "messages": messages,
        "max_new_tokens": kwargs.get("max_new_tokens", 1024),
        "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
        "stream": False,
        "temperature": kwargs.get("temperature", 0.5),
        "top_p ": kwargs.get("top_p", 0.25),
        "user": "test-video",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_secret_key,
    }

    response = requests.post(sensenova_url, headers=headers, json=data)
    try:
        response_body = json.loads(response.text)
        resp = response_body["data"]["choices"][0]["message"]
    except:
        resp = response.text
    return resp


def prepare_message_for_vllm(content_messages):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}


def call_qwen2vl(messages, model):
    video_messages, video_kwargs = prepare_message_for_vllm(messages)

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model=model,
        messages=video_messages,
        extra_body={
            "mm_processor_kwargs": video_kwargs
        }
    )
    resp = chat_response.choices[0].message.content
    return resp


def call_internvl(messages, model, **kwargs):
    api_key = 'eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIyNjIwNjQ0NSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc0MjM2OTg4OSwiY2xpZW50SWQiOiJlYm1ydm9kNnlvMG5semFlazF5cCIsInBob25lIjoiMTc4MTY4NjIyOTUiLCJvcGVuSWQiOm51bGwsInV1aWQiOiJlMjBlOGNhNi04YWUzLTRjYzEtYmUyNC00MDE0OTk2MzEzZjciLCJlbWFpbCI6IiIsImV4cCI6MTc1NzkyMTg4OX0.SxYjKjbNBWm4rlhkzSuKl_7Lf8hsQS0d357IhF8wfSCoHu1AGjdkDfdbEAKm5z2Kbd1Ccjhvtsm6ORhqIKFNsQ'
    url = 'https://chat.intern-ai.org.cn/api/v1/chat/completions'
    header = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": messages,
        "n": 1,
        "temperature": kwargs.get("temperature", 0.8),
        "top_p ": kwargs.get("top_p", 0.9),
    }

    res = requests.post(url, headers=header, data=json.dumps(data))
    print(res.status_code)
    print(res.json())
    print(res.json()["choices"][0]['message']["content"])
    resp = res.choices[0].message.content
    return resp


def call_internvl_local(messages, model, **kwargs):
    client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
    model_name = client.models.list().data[0].id
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=kwargs.get("temperature", 0.8),
        top_p=kwargs.get("top_p", 0.9),
    )
    resp = response.choices[0].message.content

    return resp


