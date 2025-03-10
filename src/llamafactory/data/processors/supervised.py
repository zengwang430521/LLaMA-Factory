# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX, DO_RESPONSE_TOKEN, NO_RESPONSE_TOKEN, VIDEO_PLACEHOLDER, FRAME_END_TOKEN, FRAME_PAD_TOKEN
from .processor_utils import greedy_knapsack, infer_seqlen
from copy import deepcopy


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)


def _encode_supervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int]]:
    messages = template.mm_plugin.process_messages(prompt + response, images, videos, processor)
    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, videos, tokenizer, processor)
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    total_length = len(input_ids) + (1 if template.efficient_eos else 0)
    if mask_history:
        encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= cutoff_len:
            break

        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if train_on_prompt:
            source_label = source_ids
        elif template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != 0:  # train on the last turn only
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        if mask_history:  # reversed sequences
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    return input_ids, labels


from ..data_utils import Role
def _encode_supervised_stream_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int]]:

    # import pdb; pdb.set_trace()
    # print('Debug: 产生input_ids, labels, stream_labels')

    # 判断视频应该怎么分段
    video_time_segs = []
    for message in prompt + response:
        content = message["content"]
        if VIDEO_PLACEHOLDER in content:
            time = message['time']
            for i in range(0, len(time), 2):
                video_time_segs.append([time[i], time[i + 1]])

    messages = template.mm_plugin.process_messages(prompt + response, images, videos, processor)

    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, videos, tokenizer, processor)

    # TODO: format 应该放在别的地方，先暂时放在这里了
    # TODO: 暂时采用粗暴的后截断，和LLAMA_FACTORY默认的截断方式不一致
    # import pdb; pdb.set_trace()
    # print('Debug: 产生input_ids, labels, stream_labels')
    assert not template.efficient_eos

    system = system or template.default_system

    # 用于stream_head回复时机的训练，
    video_pad_id = tokenizer.encode('<|video_pad|>')[0]    # 普通的 video token
    frame_end_id = tokenizer.encode(NO_RESPONSE_TOKEN)[0]    # 每一帧的最后一个 video token
    frame_response_id = tokenizer.encode(DO_RESPONSE_TOKEN)[0]   # 回复前的最后一帧的最后1个 video token

    stream_labels = [IGNORE_INDEX] * len(labels)
    for i, message in enumerate(messages):
        elements = []
        if i == 0:
            elements += template.format_prefix.apply()
            if system or tools:
                tool_text = template.format_tools.apply(content=tools)[0] if tools else ""
                elements += template.format_system.apply(content=(system + tool_text))

        elements_stream = deepcopy(elements)
        if message["role"] in [Role.USER.value, Role.OBSERVATION.value]:
            if message["role"] == Role.USER.value:
                elements += template.format_user.apply(content=message["content"], idx=str(i // 2))
                elements_stream += template.format_user.apply(content=message["content_stream"], idx=str(i // 2))
            elif message["role"] == Role.OBSERVATION.value:
                elements += template.format_observation.apply(content=message["content"])
                elements_stream += template.format_observation.apply(content=message["content_stream"])
            encode_elements = template._convert_elements_to_ids(tokenizer, elements)
            input_ids += encode_elements
            labels += [IGNORE_INDEX] * len(encode_elements)

            # 用于训练回复时机
            encode_elements_stream = template._convert_elements_to_ids(tokenizer, elements_stream)
            for token_id in encode_elements_stream:
                if token_id == frame_response_id:
                    stream_labels.append(1)
                elif token_id == frame_end_id:
                    stream_labels.append(0)
                elif token_id == video_pad_id:
                    stream_labels.append(IGNORE_INDEX)  # 不监督
                    # stream_labels.append(0)             # 监督
                else:
                    stream_labels.append(IGNORE_INDEX)
            t = 0

        elif message["role"] == Role.ASSISTANT.value:
            # 现在的写法非常死板，如果elements中本身有内容，表示是特殊情况,需要额外处理
            assert len(elements) == 0
            elements += template.format_assistant.apply(content=message["content"])
            prefix = elements[:1]
            content = elements[1:]
            encoded_prefix = template._convert_elements_to_ids(tokenizer, prefix)
            encoded_content = template._convert_elements_to_ids(tokenizer, content)
            input_ids += encoded_prefix + encoded_content
            if mask_history and i < len(messages) - 1:
                labels += [IGNORE_INDEX] * len(encoded_prefix + encoded_content)
            else:
                labels += [IGNORE_INDEX] * len(encoded_prefix) + encoded_content

            # assistant 部分没有视频，不用训练stream_head
            stream_labels += [IGNORE_INDEX] * len(encoded_prefix + encoded_content)

        elif message["role"] == Role.FUNCTION.value:
            raise NotImplementedError("Not implemented role:{}".format(message["role"]))
        else:
            raise NotImplementedError("Unexpected role: {}".format(message["role"]))

    assert len(input_ids) == len(labels) and len(input_ids) == len(stream_labels)
    if len(input_ids) > cutoff_len:
        input_ids = input_ids[:cutoff_len]
        labels = labels[:cutoff_len]
        stream_labels = stream_labels[:cutoff_len]

    return input_ids, labels, stream_labels, video_time_segs


def _encode_supervised_stream_example_v2(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int]]:
    # <video> <text> <video> 交错的版本
    # stream labels 应该在每帧最后的 video token 和问题最后的文本token上计算

    # import pdb; pdb.set_trace()
    # print('Debug V2: 产生input_ids, labels, stream_labels')

    # 判断视频应该怎么分段
    video_time_segs = []
    for message in prompt + response:
        content = message["content"]
        if VIDEO_PLACEHOLDER in content:
            time = message['time']
            for i in range(0, len(time), 2):
                video_time_segs.append([time[i], time[i+1]])

    # import pdb; pdb.set_trace()
    messages, frame_idxs, frame_times, video_grid_thw = template.mm_plugin.process_messages(prompt + response, images, videos, processor)
    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, videos, tokenizer, processor)

    # TODO: format 应该放在别的地方，先暂时放在这里了
    # TODO: 暂时采用粗暴的后截断，和LLAMA_FACTORY默认的截断方式不一致
    # import pdb; pdb.set_trace()
    # print('Debug: 产生input_ids, labels, stream_labels')
    assert not template.efficient_eos

    system = system or template.default_system

    # 用于stream_head回复时机的训练，
    video_pad_id = tokenizer.encode('<|video_pad|>')[0]    # 普通的 video token
    frame_end_id = tokenizer.encode(FRAME_END_TOKEN)[0]    # 每一帧的最后一个 video token
    text_end_id = tokenizer.encode('<|im_end|>')[0]
    vision_end_id = tokenizer.encode('<|vision_end|>')[0]

    stream_labels = [IGNORE_INDEX] * len(labels)
    for i, message in enumerate(messages):
        elements = []
        if i == 0:
            elements += template.format_prefix.apply()
            if system or tools:
                tool_text = template.format_tools.apply(content=tools)[0] if tools else ""
                elements += template.format_system.apply(content=(system + tool_text))

        elements_stream = deepcopy(elements)
        if message["role"] in [Role.USER.value, Role.OBSERVATION.value]:
            if message["role"] == Role.USER.value:
                elements += template.format_user.apply(content=message["content"], idx=str(i // 2))
                elements_stream += template.format_user.apply(content=message["content_stream"], idx=str(i // 2))
            elif message["role"] == Role.OBSERVATION.value:
                elements += template.format_observation.apply(content=message["content"])
                elements_stream += template.format_observation.apply(content=message["content_stream"])
            encode_elements = template._convert_elements_to_ids(tokenizer, elements)
            input_ids += encode_elements
            labels += [IGNORE_INDEX] * len(encode_elements)

            # 用于训练回复时机
            need_response = False
            if i + 1 < len(messages):
                next_message = messages[i + 1]
                need_response = (next_message["role"] == Role.ASSISTANT.value)
            encode_elements_stream = template._convert_elements_to_ids(tokenizer, elements_stream)

            judge_spots = []
            for idx, token_idx in enumerate(encode_elements_stream):
                if token_idx == frame_end_id:
                    # 每一帧的最后一个vision token是判别点
                    judge_spots.append(idx)
                elif token_idx == text_end_id and encode_elements_stream[idx-1] != vision_end_id:
                    # 视频后有文本内容的话，最后的<|im_end|>是判别点
                    judge_spots.append(idx)

            # 最后1个判别点根据情况判断，之前的其他判别点不能回复
            tmp_stream_labels = [IGNORE_INDEX] * len(encode_elements_stream)
            for idx in judge_spots[:-1]:
                tmp_stream_labels[idx] = 0
            tmp_stream_labels[judge_spots[-1]] = int(need_response)
            stream_labels += tmp_stream_labels
            t = 0

        elif message["role"] == Role.ASSISTANT.value:
            # 现在的写法非常死板，如果elements中本身有内容，表示是特殊情况,需要额外处理
            assert len(elements) == 0
            elements += template.format_assistant.apply(content=message["content"])
            prefix = elements[:1]
            content = elements[1:]
            encoded_prefix = template._convert_elements_to_ids(tokenizer, prefix)
            encoded_content = template._convert_elements_to_ids(tokenizer, content)
            input_ids += encoded_prefix + encoded_content
            if mask_history and i < len(messages) - 1:
                labels += [IGNORE_INDEX] * len(encoded_prefix + encoded_content)
            else:
                labels += [IGNORE_INDEX] * len(encoded_prefix) + encoded_content

            # assistant 部分没有视频，不用训练stream_head
            stream_labels += [IGNORE_INDEX] * len(encoded_prefix + encoded_content)

        elif message["role"] == Role.FUNCTION.value:
            raise NotImplementedError("Not implemented role:{}".format(message["role"]))
        else:
            raise NotImplementedError("Unexpected role: {}".format(message["role"]))

    assert len(input_ids) == len(labels) and len(input_ids) == len(stream_labels)
    import pdb; pdb.set_trace()
    if len(input_ids) > cutoff_len:
        input_ids = input_ids[:cutoff_len]
        labels = labels[:cutoff_len]
        stream_labels = stream_labels[:cutoff_len]

    return input_ids, labels, stream_labels, frame_idxs, frame_times, video_grid_thw




def _encode_supervised_stream_example_v3(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int]]:
    # <video> <text> <video> 交错的版本
    # stream labels 应该在<|im_end|>上计算。
    # 所以视频中每帧后都需要加上<|vision_end|><|im_end|>
    # 需要mask掉额外新加的token

    # import pdb; pdb.set_trace()
    # print('Debug V3: 产生input_ids, labels, stream_labels')

    # 判断视频应该怎么分段
    video_time_segs = []
    for message in prompt + response:
        content = message["content"]
        if VIDEO_PLACEHOLDER in content:
            time = message['time']
            for i in range(0, len(time), 2):
                video_time_segs.append([time[i], time[i+1]])

    # import pdb; pdb.set_trace()
    messages, frame_idxs, frame_times, video_grid_thw = template.mm_plugin.process_messages(prompt + response, images, videos, processor)
    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, videos, tokenizer, processor)
    masks = [1] * len(input_ids)

    # TODO: format 应该放在别的地方，先暂时放在这里了
    # TODO: 暂时采用粗暴的后截断，和LLAMA_FACTORY默认的截断方式不一致
    # import pdb; pdb.set_trace()
    # print('Debug: 产生input_ids, labels, stream_labels')
    assert not template.efficient_eos

    system = system or template.default_system

    # 用于stream_head回复时机的训练，
    frame_pad_id = tokenizer.encode(FRAME_PAD_TOKEN)[0]    # 每一帧的最后一个 video token
    judge_id = tokenizer.encode('<|im_end|>')[0]

    stream_labels = [IGNORE_INDEX] * len(labels)
    for i, message in enumerate(messages):
        elements = []
        if i == 0:
            elements += template.format_prefix.apply()
            if system or tools:
                tool_text = template.format_tools.apply(content=tools)[0] if tools else ""
                elements += template.format_system.apply(content=(system + tool_text))

        elements_stream = deepcopy(elements)
        if message["role"] in [Role.USER.value, Role.OBSERVATION.value]:
            if message["role"] == Role.USER.value:
                elements += template.format_user.apply(content=message["content"], idx=str(i // 2))
                elements_stream += template.format_user.apply(content=message["content_stream"], idx=str(i // 2))
            elif message["role"] == Role.OBSERVATION.value:
                elements += template.format_observation.apply(content=message["content"])
                elements_stream += template.format_observation.apply(content=message["content_stream"])
            encode_elements = template._convert_elements_to_ids(tokenizer, elements)
            input_ids += encode_elements
            labels += [IGNORE_INDEX] * len(encode_elements)

            # 用于训练回复时机
            need_response = False
            if i + 1 < len(messages):
                next_message = messages[i + 1]
                need_response = (next_message["role"] == Role.ASSISTANT.value)
            encode_elements_stream = template._convert_elements_to_ids(tokenizer, elements_stream)

            # encode_elements_stream 用来生成mask
            # 判定点直接用 encode_elements 找
            judge_spots = []
            for idx, token_idx in enumerate(encode_elements):
                if token_idx == judge_id:
                    judge_spots.append(idx)

            # 最后1个判别点根据情况判断，之前的其他判别点不能回复
            tmp_stream_labels = [IGNORE_INDEX] * len(encode_elements_stream)
            for idx in judge_spots[:-1]:
                tmp_stream_labels[idx] = 0
            tmp_stream_labels[judge_spots[-1]] = int(need_response)
            stream_labels += tmp_stream_labels

            mask = [0  if t == frame_pad_id else 1 for t in encode_elements_stream]
            masks += mask

        elif message["role"] == Role.ASSISTANT.value:
            # 现在的写法非常死板，如果elements中本身有内容，表示是特殊情况,需要额外处理
            assert len(elements) == 0
            elements += template.format_assistant.apply(content=message["content"])
            prefix = elements[:1]
            content = elements[1:]
            encoded_prefix = template._convert_elements_to_ids(tokenizer, prefix)
            encoded_content = template._convert_elements_to_ids(tokenizer, content)
            input_ids += encoded_prefix + encoded_content
            if mask_history and i < len(messages) - 1:
                labels += [IGNORE_INDEX] * len(encoded_prefix + encoded_content)
            else:
                labels += [IGNORE_INDEX] * len(encoded_prefix) + encoded_content

            # assistant 部分没有视频，不用训练stream_head
            stream_labels += [IGNORE_INDEX] * len(encoded_prefix + encoded_content)
            masks += [1] * len(encoded_prefix + encoded_content)

        elif message["role"] == Role.FUNCTION.value:
            raise NotImplementedError("Not implemented role:{}".format(message["role"]))
        else:
            raise NotImplementedError("Unexpected role: {}".format(message["role"]))

    assert len(input_ids) == len(labels) and len(input_ids) == len(stream_labels)
    import pdb; pdb.set_trace()
    if len(input_ids) > cutoff_len:
        input_ids = input_ids[:cutoff_len]
        labels = labels[:cutoff_len]
        stream_labels = stream_labels[:cutoff_len]
        masks = masks[:cutoff_len]

    return input_ids, labels, stream_labels, frame_idxs, frame_times, video_grid_thw, masks




def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.

    model_inputs = defaultdict(list)
    # print('debug')
    for i in range(len(examples["_prompt"])):
        if data_args.template == 'qwen2_vl_stream':
            # qwen2_vl_stream 对话数据不进行验证, 并且需要额外的stream_labels
            input_ids, labels, stream_labels, video_time_segs = _encode_supervised_stream_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                cutoff_len=data_args.cutoff_len,
                train_on_prompt=data_args.train_on_prompt,
                mask_history=data_args.mask_history,
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["stream_labels"].append(stream_labels)
            model_inputs["video_time_segs"].append(video_time_segs)

        elif data_args.template == 'qwen2_vl_stream_v2':
            # qwen2_vl_stream 对话数据不进行验证, 并且需要额外的stream_labels
            # 数据集中少量视频文件有问题，放弃这些数据
            try:
                input_ids, labels, stream_labels, frame_idxs, frame_times, video_grid_thw = _encode_supervised_stream_example_v2(
                    prompt=examples["_prompt"][i],
                    response=examples["_response"][i],
                    system=examples["_system"][i],
                    tools=examples["_tools"][i],
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    template=template,
                    tokenizer=tokenizer,
                    processor=processor,
                    cutoff_len=data_args.cutoff_len,
                    train_on_prompt=data_args.train_on_prompt,
                    mask_history=data_args.mask_history,
                )
                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append([1] * len(input_ids))
                model_inputs["labels"].append(labels)
                model_inputs["images"].append(examples["_images"][i])
                model_inputs["videos"].append(examples["_videos"][i])
                model_inputs["stream_labels"].append(stream_labels)
                model_inputs["frame_idxs"].append(frame_idxs)
                model_inputs["frame_times"].append(frame_times)
                model_inputs["video_grid_thw"].append(video_grid_thw)
            except:
                print(f'Skip broken data!!!:{examples["_videos"][i]}.')
                # import pdb; pdb.set_trace()
                # input_ids, labels, stream_labels, video_time_segs = _encode_supervised_stream_example_v2(
                #     prompt=examples["_prompt"][i],
                #     response=examples["_response"][i],
                #     system=examples["_system"][i],
                #     tools=examples["_tools"][i],
                #     images=examples["_images"][i] or [],
                #     videos=examples["_videos"][i] or [],
                #     template=template,
                #     tokenizer=tokenizer,
                #     processor=processor,
                #     cutoff_len=data_args.cutoff_len,
                #     train_on_prompt=data_args.train_on_prompt,
                #     mask_history=data_args.mask_history,
                # )


        elif data_args.template == 'qwen2_vl_stream_v3':
            # qwen2_vl_stream 对话数据不进行验证, 并且需要额外的stream_labels
            # 数据集中少量视频文件有问题，放弃这些数据
            try:
                input_ids, labels, stream_labels, frame_idxs, frame_times, video_grid_thw, masks = _encode_supervised_stream_example_v3(
                    prompt=examples["_prompt"][i],
                    response=examples["_response"][i],
                    system=examples["_system"][i],
                    tools=examples["_tools"][i],
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    template=template,
                    tokenizer=tokenizer,
                    processor=processor,
                    cutoff_len=data_args.cutoff_len,
                    train_on_prompt=data_args.train_on_prompt,
                    mask_history=data_args.mask_history,
                )
                model_inputs["input_ids"].append(input_ids)
                # model_inputs["attention_mask"].append([1] * len(input_ids))
                model_inputs["labels"].append(labels)
                model_inputs["images"].append(examples["_images"][i])
                model_inputs["videos"].append(examples["_videos"][i])
                model_inputs["stream_labels"].append(stream_labels)
                model_inputs["frame_idxs"].append(frame_idxs)
                model_inputs["frame_times"].append(frame_times)
                model_inputs["video_grid_thw"].append(video_grid_thw)
                model_inputs["attention_mask"].append(masks)
            except:
                print(f'Skip broken data!!!:{examples["_videos"][i]}.')



        else:
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = _encode_supervised_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                cutoff_len=data_args.cutoff_len,
                train_on_prompt=data_args.train_on_prompt,
                mask_history=data_args.mask_history,
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])

    # import pdb; pdb.set_trace()
    return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # TODO: use `position_ids` to achieve packing
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`

    valid_num = 0
    batch_input_ids, batch_labels, batch_images, batch_videos = [], [], [], []
    lengths = []
    length2indexes = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len - 1,  # reserved for the padding token
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        length = len(input_ids)
        if length > data_args.cutoff_len:
            logger.warning_rank0(f"Dropped lengthy example with length {length} > {data_args.cutoff_len}.")
        else:
            lengths.append(length)
            length2indexes[length].append(valid_num)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_images.append(examples["_images"][i] or [])
            batch_videos.append(examples["_videos"][i] or [])
            valid_num += 1

    model_inputs = defaultdict(list)
    knapsacks = greedy_knapsack(lengths, data_args.cutoff_len - 1)  # reserved for the padding token
    for knapsack in knapsacks:
        packed_input_ids, packed_attention_masks, packed_labels = [], [], []
        packed_images, packed_videos = [], []
        for i, length in enumerate(knapsack):
            index = length2indexes[length].pop()
            packed_input_ids += batch_input_ids[index]
            packed_labels += batch_labels[index]
            packed_images += batch_images[index]
            packed_videos += batch_videos[index]
            if data_args.neat_packing:
                packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
            else:
                packed_attention_masks += [1] * len(batch_input_ids[index])

        if len(packed_input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(packed_input_ids)
            packed_input_ids += [tokenizer.pad_token_id] * pad_length
            packed_labels += [IGNORE_INDEX] * pad_length
            if data_args.neat_packing:
                packed_attention_masks += [0] * pad_length
            else:
                packed_attention_masks += [1] * pad_length  # more efficient flash_attn

        if len(packed_input_ids) != data_args.cutoff_len:
            raise ValueError("The length of packed example should be identical to the cutoff length.")

        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["labels"].append(packed_labels)
        model_inputs["images"].append(packed_images or None)
        model_inputs["videos"].append(packed_videos or None)

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print(f"labels:\n{tokenizer.decode(valid_labels, skip_special_tokens=False)}")
