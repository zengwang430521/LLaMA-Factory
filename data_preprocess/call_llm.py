# 调用llm和vllm
import os
import io
import PIL
import base64
import openai
from os.path import exists, join
import numpy as np
import jwt
import time
import requests
import json
import cv2

lmdploy_model_urls = {
    # 'internvl2_26b': os.getenv('INTERNVL2_26B_URL', 'http://127.0.0.1:23333/v1'),
    'qwen2_vl_7b_instruct': os.getenv('QWEN2_VL_7B_URL', 'http://127.0.0.1:23334/v1')
    # 'qwen2_vl_7b_instruct': os.getenv('QWEN2_VL_7B_URL', 'http://103.237.29.227:8193/v1')
}
lmdploy_models = list(lmdploy_model_urls.keys())

sglang_model_urls = {
    'llava_onevision_chat_72b': os.getenv('LLAVA_ONEVISION_CHAT_72B_URL', 'http://127.0.0.1:30000/v1')
}
sglang_models = list(sglang_model_urls.keys())

sensenova_url = "https://api.stage.sensenova.cn/v1/llm/chat-completions"
sensenova_models = ['SenseChat-Vision-102b-stage']
ak = "2TNIeaXuVk9eF5TuJDujX7extFJ"  # 填写您的ak
sk = "cX4pGI06hkgJRBnv3cYH5bF8aKnYRD59"  # 填写您的sk


# sensenova_url = "https://api.sensenova.cn/v1/llm/chat-completions"
# sensenova_models = ['SenseChat-Vision']
# ak = "C563364C54234B4991B40F250B307974"
# sk = "79F53ED63ADB4DCE8097542DA3E9BDD8"



def pil_image_to_base64(image):
    """
    PIL.Image.Image 转换为 base64 格式的字符串

    Args:
        image (PIL.Image.Image): 图片

    Returns:
        image_base64 (str): base64格式的字符串
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # 选择合适的格式（如JPEG或PNG）
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return image_base64


def image_content_to_base64(image):
    """
    把对话中的图片转换为 base64 格式的字符串

    Args:
        image (str/np.array/PIL.Image.Image): 图片

    Returns:
        image_base64 (str): base64格式的字符串
    """
    if isinstance(image, str):
        if exists(image):
            # 文件路径
            image_base64 = base64.b64encode(open(image, "rb").read()).decode("utf-8")
        else:
            # 已经是base64了
            image_base64 = image
    elif isinstance(image, PIL.Image.Image):
        image_base64 = pil_image_to_base64(image)
    elif isinstance(image, np.ndarray):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_base64 = pil_image_to_base64(PIL.Image.fromarray(image_rgb))
    else:
        raise NotImplementedError(f'Unsupported image type: {type(image)}')
    return image_base64


def call_lmdeploy_model(messages, model, **kwargs):
    """
    调用通过lmdeploy部署的模型

    Args:
        messages (list): messages的元素格式为{'role': str, 'content': str(text) or tuple (image, text)},
            image 支持 文件名，PIL.Image.Image, base64_str 格式。
        model (str): 要调用的模型名称
        kwargs (dict): llm generation params

    Returns:
        resp (str): 模型回复
    """

    # 首先处理messages的格式
    new_messages = []
    for msg in messages:
        role, content = msg['role'], msg['content']
        if isinstance(content, tuple) or isinstance(content, list):
            images, text = content
            if text is None:
                text = ''
            new_msg = {
                "role": role,
                "content": [
                    {"type": "text", "text": text},
                ]
            }
            if not isinstance(images, list):
                images = [images]
            for image in images:
                image_base64 = image_content_to_base64(image)
                image_url = f"data:image/jpeg;base64,{image_base64}"
                new_msg["content"].append({"type": "image_url", "image_url": {"url": image_url, "detail": "high"}})
        elif isinstance(content, str):
            new_msg = {"role": role, "content": content}
        else:
            raise TypeError(f'Unsupproted content type: {type(content)}')
        new_messages.append(new_msg)

    base_url = lmdploy_model_urls[model]
    client = openai.OpenAI(
        api_key='YOUR_API_KEY',
        base_url=base_url
    )
    # lmdeploy 部署下，只有一个model，model_id不重要
    model_id = client.models.list().data[0].id
    response = client.chat.completions.create(
        model=model_id,
        messages=new_messages,
        **kwargs
    )
    resp = response.choices[0].message.content
    return resp


def call_sglang_model(messages, model, **kwargs):
    """
    调用通过sglang部署的模型

    Args:
        messages (list): messages的元素格式为{'role': str, 'content': str(text) or tuple (image, text)},
            image 支持 文件名，PIL.Image.Image, base64_str 格式。
        model (str): 要调用的模型名称
        kwargs (dict): llm generation params

    Returns:
        resp (str): 模型回复
    """

    num_images = 0
    for msg in messages:
        content = msg['content']
        if isinstance(content, tuple) or isinstance(content, list):
            images, text = content
            if not isinstance(images, list):
                num_images += 1
            else:
                num_images += len(images)

    # 首先处理messages的格式
    new_messages = []
    for msg in messages:
        role, content = msg['role'], msg['content']
        if isinstance(content, tuple) or isinstance(content, list):
            images, text = content
            if text is None:
                text = ''
            new_msg = {
                "role": role,
                "content": [
                    {"type": "text", "text": text},
                ]
            }
            if not isinstance(images, list):
                images = [images]

            for image in images:
                image_base64 = image_content_to_base64(image)
                image_url = f"data:image/jpeg;base64,{image_base64}"
                image_content = {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
                if num_images > 1:
                    image_content['modalities'] = "multi-images"
                new_msg["content"].append(image_content)
        elif isinstance(content, str):
            new_msg = {"role": role, "content": content}
        else:
            raise TypeError(f'Unsupproted content type: {type(content)}')
        new_messages.append(new_msg)

    base_url = sglang_model_urls[model]

    client = openai.OpenAI(
        api_key='None',
        base_url=base_url
    )
    # lmdeploy 部署下，只有一个model，model_id不重要
    model_id = client.models.list().data[0].id
    response = client.chat.completions.create(
        model=model_id,
        messages=new_messages,
        **kwargs
    )
    resp = response.choices[0].message.content
    return resp


# 生成鉴权token
def encode_jwt_token(ak, sk):
    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800,  # 填写您期望的有效时间，此处示例代表当前时间+30分钟
        "nbf": int(time.time()) - 120,  # 填写您期望的生效时间，此处示例代表当前时间-5秒
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token


def call_sensenova_model(messages, model, **kwargs):
    """
    调用通过sensenova上部署的模型

    Args:
        messages (list): messages的元素格式为{'role': str, 'content': str(text) or tuple (image, text)},
            image 支持 文件名，PIL.Image.Image, base64_str 格式。
        model (str): 要调用的模型名称
        kwargs (dict): llm generation params

    Returns:
        resp (str): 模型回复
    """

    # 首先处理messages的格式
    new_messages = []
    for msg in messages:
        role, content = msg['role'], msg['content']
        if isinstance(content, tuple) or isinstance(content, list):
            images, text = content
            if text is None:
                text = ''
            new_msg = {"role": role, "content": [{"type": "text", "text": text}]}
            if not isinstance(images, list):
                images = [images]
            for image in images:
                image_base64 = image_content_to_base64(image)
                new_msg["content"].append({"type": "image_base64", "image_base64": image_base64})
        elif isinstance(content, str):
            new_msg = {"role": role, "content": [{"type": "text", "text": content}]}
        else:
            raise TypeError(f'Unsupproted content type: {type(content)}')
        new_messages.append(new_msg)

    api_secret_key = encode_jwt_token(ak, sk)
    data = {
        "messages": new_messages,
        "temperature": kwargs.get("temperature", 0.5),
        "max_new_tokens": kwargs.get("max_new_tokens", 1024),
        "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
        "top_p ": kwargs.get("top_p", 0.25),
        "model": model,
        "stream": False,
        "user": "test-xhs",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_secret_key,
    }

    response = requests.post(sensenova_url, headers=headers, json=data)
    response_body = json.loads(response.text)
    resp = response_body["data"]["choices"][0]["message"]
    return resp


def call_chatapi_model(messages, model, **kwargs):
    # ak = "196FFD00B6184227B65B3D92C01A8724"
    # sk = "DD1D004D80834448B276F125F8310F2A"
    # api_secret_key = encode_jwt_token(ak, sk)

    api_secret_key = 'sk-vpwnt8A6wKnCKd5GKYnnBcjJ63ypiiIl'

    data = {
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.5),
        "max_new_tokens": kwargs.get("max_new_tokens", 2048),
        "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
        "top_p ": kwargs.get("top_p", 0.25),
        "model": model,
        "plugins": {}
        # "stream": False,
        # "user": "test-xhs",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_secret_key,
    }

    response = requests.post("https://chatapi.sensenova.cn/v1/llm/chat-completions", headers=headers, json=data)
    response_body = json.loads(response.text)
    resp = response_body["data"]["choices"][0]["message"]
    return resp


def call_llm_completion(messages, model, **kwargs):
    """
    调用llm/vllm

    Args:
        messages (list): 列表中元素格式：{'role': str, 'content': str(text) or tuple (file, text)}
        model (str): 要调用的模型名称
        kwargs (dict): llm generation params

    Returns:
        resp (str): 模型回复
    """

    if model in lmdploy_models:
        resp = call_lmdeploy_model(messages, model, **kwargs)
    elif model in sensenova_models:
        resp = call_sensenova_model(messages, model, **kwargs)
    elif model in sglang_models:
        resp = call_sglang_model(messages, model, **kwargs)
    else:
        raise ValueError(f'Unknown model name: {model}')
    # print(f'response: {resp}')
    return resp


def call_silionflow_model(messages, model, **kwargs):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    api_secret_key = 'sk-znwnezzbthpkjboanoecocwnuklcstqfzjjpjizyveodvqiw'
    payload = {
        "model": model,
        "messages": messages,
        "stream": kwargs.get("stream", False),
        "max_tokens": kwargs.get("max_tokens", 2048),
        "stop": ["null"],
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 0.7),
        "top_k": kwargs.get("top_k", 50),
        "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
        "n": 1,
        "response_format": {"type": "text"},
    }
    headers = {
        "Authorization": "Bearer " + api_secret_key,
        "Content-Type": "application/json"
    }
    response = requests.request("POST", url, json=payload, headers=headers)
    print(response.text)
    return response



from openai import AzureOpenAI
def call_openai_model(messages, model="gpt-4o", **kwargs):
    endpoint = os.getenv("ENDPOINT_URL", "https://jinsheng-us-e.openai.azure.com/")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "6351e2a233704293a9d65c422d3785d2")

    # 使用基于密钥的身份验证来初始化 Azure OpenAI 客户端
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-05-01-preview",
    )

    # 生成完成
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    res = completion.dict()
    response = res['choices'][0]['message']['content']

    prompt_tokens = res["usage"]["prompt_tokens"]
    completion_tokens = res["usage"]["completion_tokens"]
    cost = prompt_tokens * 2.50 * 1e-6 + completion_tokens * 10.00 * 1e-6

    result = {
        "response": response,
        "cost": cost
    }
    return result










if __name__ == '__main__':
    messages = [{'role': 'user', 'content': '你是谁？'}]
    # messages = [{'role': 'user', 'content': [['img_path1', 'img_path2'], '描述两张图的区别']}]

    # model = 'llava_onevision_chat_72b'
    # model = 'qwen2_vl_7b_instruct'
    model = 'SenseChat-Vision-102b-stage'

    resp = call_llm_completion(messages, model)
    print(resp)

