# Copyright 2025 the LlamaFactory team.
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen
from copy import deepcopy
from ...extras.constants import (
    IGNORE_INDEX, VIDEO_PLACEHOLDER, IMAGE_PLACEHOLDER,
    DO_RESPONSE_TOKEN, NO_RESPONSE_TOKEN, NULL_RESPONSE_TOKEN,
    FRAME_END_TOKEN, FRAME_PAD_TOKEN)
from  ..data_utils import Role


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


def get_image_video_grid_num(messages):
    num_image, num_video, num_video_grid = 0, 0, 0
    messages = deepcopy(messages)
    for message in messages:
        content = message["content"]
        num_image += content.count(IMAGE_PLACEHOLDER)
        num_video += content.count(VIDEO_PLACEHOLDER)

        while "<video><+><video>" in content:
            content = content.replace("<video><+><video>", "<video>", 1)
        num_video_grid += content.count(VIDEO_PLACEHOLDER)

    return num_image, num_video, num_video_grid


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def _encode_data_example_stream_v5(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ):
        cutoff_len = self.data_args.cutoff_len
        mask_history = self.data_args.mask_history
        template = self.template

        messages, frame_idxs, frame_times, video_grid_thw, fps_per_video = \
            self.template.mm_plugin.process_messages(
                prompt + response,
                images,
                videos,
                audios,
                self.processor
            )
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        masks = [1] * len(input_ids)
        stream_labels = [IGNORE_INDEX] * len(labels)

        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        # TODO: format 应该放在别的地方，先暂时放在这里了
        # TODO: 暂时采用粗暴的后截断，和LLAMA_FACTORY默认的截断方式不一致
        # import pdb; pdb.set_trace()
        # print('Debug: 产生input_ids, labels, stream_labels')
        assert not self.template.efficient_eos
        system = system or self.template.default_system

        # 用于stream_head回复时机的训练，
        frame_pad_id = self.tokenizer.encode(FRAME_PAD_TOKEN)[0]  # pad的需要mask的token
        judge_id = self.tokenizer.encode('<|im_end|>')[0]  # stream 判别点
        do_response_id = self.tokenizer.encode(DO_RESPONSE_TOKEN)[0]
        no_response_id = self.tokenizer.encode(NO_RESPONSE_TOKEN)[0]
        null_response_id = self.tokenizer.encode(NULL_RESPONSE_TOKEN)[0]

        reserved_message_num = 0
        for i, message in enumerate(messages):
            total_len = len(input_ids)

            '''超出长度就不要了'''
            if total_len >= cutoff_len:
                break

            elements = []
            if i == 0:
                elements += template.format_prefix.apply()
                if system or tools:
                    tool_text = template.format_tools.apply(content=tools)[0] if tools else ""
                    elements += template.format_system.apply(content=(system + tool_text))

            elements_mask = deepcopy(elements)
            elements_stream = deepcopy(elements)

            if message["role"] in [Role.USER.value, Role.OBSERVATION.value]:
                if message["role"] == Role.USER.value:
                    elements += template.format_user.apply(content=message["content"], idx=str(i // 2))
                    elements_mask += template.format_user.apply(content=message["content_mask"], idx=str(i // 2))
                    elements_stream += template.format_user.apply(content=message["content_stream"], idx=str(i // 2))

                elif message["role"] == Role.OBSERVATION.value:
                    elements += template.format_observation.apply(content=message["content"])
                    elements_mask += template.format_observation.apply(content=message["content_mask"])
                    elements_stream += template.format_observation.apply(content=message["content_stream"])

                # 实际的输入
                encode_elements = template._convert_elements_to_ids(self.tokenizer, elements)

                # attention mask
                encode_elements_mask = template._convert_elements_to_ids(self.tokenizer, elements_mask)
                mask = [0 if t == frame_pad_id else 1 for t in encode_elements_mask]

                # 用于训练回复时机
                encode_elements_stream = template._convert_elements_to_ids(self.tokenizer, elements_stream)
                tmp_stream_labels = [
                    1 if t == do_response_id else 0 if t == no_response_id else IGNORE_INDEX
                    for t in encode_elements_stream
                ]

                ignore_end_stream = message.get("ignore_end_stream", None) or False
                if not ignore_end_stream:
                    # 最后一个判定点, 用 encode_elements 找
                    need_response = False
                    if i + 1 < len(messages):
                        next_message = messages[i + 1]
                        need_response = (next_message["role"] == Role.ASSISTANT.value)

                    for idx, token_idx in enumerate(encode_elements):
                        if token_idx == judge_id:
                            judge_spot = idx
                    tmp_stream_labels[judge_spot] = 1 if need_response else 0

                cur_len = len(encode_elements)
                if total_len + cur_len > cutoff_len:
                    '''
                    加上之后会超出长度。
                    因为可能有视频存在，把<|video_pad|>截断的话，get_rope_index会出问题,
                    所以不能直接做截断, 干脆放弃这轮对话
                    '''
                    break

                input_ids += encode_elements
                labels += [IGNORE_INDEX] * len(encode_elements)
                if mask_history and i < len(messages) - 3:
                    # mask_history 时，只计算最后 3 条 message 的 stream label
                    # 1. 最后3条 message 为 <Assistant> <User> <Assistant>
                    # 2. 最后3条 message 为 <User> <User> <Assistant>，可以训练模型不要马上回答。
                    stream_labels += [IGNORE_INDEX] * len(encode_elements)
                else:
                    stream_labels += tmp_stream_labels
                masks += mask
                reserved_message_num += 1
            elif message["role"] == Role.ASSISTANT.value:
                # 现在的写法非常死板，如果elements中本身有内容，表示是特殊情况,需要额外处理
                assert len(elements) == 0
                elements += template.format_assistant.apply(content=message["content"])
                prefix = elements[:1]
                content = elements[1:]
                encoded_prefix = template._convert_elements_to_ids(self.tokenizer, prefix)
                encoded_content = template._convert_elements_to_ids(self.tokenizer, content)
                encode_elements = encoded_prefix + encoded_content

                valid = message.get('valid', True)
                if valid is None:
                    valid = True

                if mask_history and i < len(messages) - 1:
                    encode_labels = [IGNORE_INDEX] * len(encoded_prefix + encoded_content)
                elif not valid:
                    # 有时候会有 fake response
                    encode_labels = [IGNORE_INDEX] * len(encoded_prefix + encoded_content)
                else:
                    encode_labels = [IGNORE_INDEX] * len(encoded_prefix) + encoded_content

                cur_len = len(encode_elements)
                if total_len + cur_len >= cutoff_len:
                    encode_elements = encode_elements[:cutoff_len - total_len]
                    encode_labels = encode_labels[:cutoff_len - total_len]

                input_ids += encode_elements
                labels += encode_labels
                # assistant 部分没有视频，不用训练stream_head
                stream_labels += [IGNORE_INDEX] * len(encode_elements)
                masks += [1] * len(encode_elements)
                reserved_message_num += 1

            elif message["role"] == Role.FUNCTION.value:
                raise NotImplementedError("Not implemented role:{}".format(message["role"]))
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

        assert len(input_ids) == len(labels) and len(input_ids) == len(stream_labels)
        return (input_ids, labels, stream_labels, frame_idxs, frame_times,
                video_grid_thw, fps_per_video, masks, reserved_message_num)

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if self.data_args.template == 'qwen2_vl_stream_v5':
                try:
                    (input_ids, labels, stream_labels, frame_idxs, frame_times,
                     video_grid_thw, fps_per_video, masks, reserved_message_num) = \
                        self._encode_data_example_stream_v5(
                            prompt=examples["_prompt"][i],
                            response=examples["_response"][i],
                            system=examples["_system"][i],
                            tools=examples["_tools"][i],
                            images=examples["_images"][i] or [],
                            videos=examples["_videos"][i] or [],
                            audios=examples["_audios"][i] or [],
                        )
                except:
                    print(f'Skip broken data!!!:{examples["_videos"][i]}.')
                    continue

                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append([1] * len(input_ids))
                model_inputs["labels"].append(labels)
                model_inputs["stream_labels"].append(stream_labels)

                messages = examples["_prompt"][i] + examples["_response"][i]
                num_image, num_video, num_video_grid = get_image_video_grid_num(messages[:reserved_message_num])

                images = examples["_images"][i]
                if images is not None:
                    images = images[:num_image]
                model_inputs["images"].append(images)

                videos = examples["_videos"][i]
                if videos is not None:
                    videos = videos[:num_video]
                model_inputs["videos"].append(videos)

                model_inputs["frame_idxs"].append(frame_idxs[:num_video])
                model_inputs["frame_times"].append(frame_times[:num_video])
                model_inputs["video_grid_thw"].append(video_grid_thw[:num_video_grid])
                model_inputs["fps_per_video"].append(fps_per_video[:num_video_grid])

                # TODO: 暂时不具备处理audio的能力
                model_inputs["audios"].append(examples["_audios"][i])

            else:

                if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                    logger.warning_rank0(
                        "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                    )
                    continue

                input_ids, labels = self._encode_data_example(
                    prompt=examples["_prompt"][i],
                    response=examples["_response"][i],
                    system=examples["_system"][i],
                    tools=examples["_tools"][i],
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                )
                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append([1] * len(input_ids))
                model_inputs["labels"].append(labels)
                model_inputs["images"].append(examples["_images"][i])
                model_inputs["videos"].append(examples["_videos"][i])
                model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs
