import math
import os.path
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from typing_extensions import override

from ..extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER, DO_RESPONSE_TOKEN, NO_RESPONSE_TOKEN, FRAME_END_TOKEN, FRAME_PAD_TOKEN
from ..extras.packages import is_pillow_available, is_pyav_available, is_transformers_version_greater_than


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_pyav_available():
    import av


if is_transformers_version_greater_than("4.45.0"):
    from transformers.models.mllama.processing_mllama import (
        convert_sparse_cross_attention_mask_to_dense,
        get_cross_attention_token_mask,
    )


if TYPE_CHECKING:
    from av.stream import Stream
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, ImageObject]
    VideoInput = str


def _get_paligemma_token_type_ids(
    imglens: Sequence[int], seqlens: Sequence[int], processor: "ProcessorMixin"
) -> List[List[int]]:
    r"""
    Gets paligemma token type ids for computing loss.

    Returns:
        batch_token_type_ids: shape (batch_size, sequence_length)
    """
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * getattr(processor, "image_seqlen")
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


class BasePlugin:
    def __init__(self, image_token: Optional[str], video_token: Optional[str]) -> None:
        self.image_token = image_token
        self.video_token = video_token
        self.expand_mm_tokens = True

    def _validate_input(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")

        if len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")

    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image.
        """
        image_resolution: int = kwargs.get("image_resolution")
        if (image.width * image.height) > image_resolution:
            resize_factor = math.sqrt(image_resolution / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        r"""
        Computes video sample frames according to fps.
        """
        video_fps: float = kwargs.get("video_fps")
        video_maxlen: int = kwargs.get("video_maxlen")
        total_frames = video_stream.frames
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return math.floor(sample_frames)

    def _regularize_images(self, images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of Images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))

        return results

    def _regularize_videos(self, videos: Sequence["VideoInput"], **kwargs) -> List[List["ImageObject"]]:
        r"""
        Regularizes videos to avoid error. Including reading, resizing and converting.
        """
        results = []
        for video in videos:
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            total_frames = video_stream.frames
            sample_frames = self._get_video_sample_frames(video_stream, **kwargs)
            sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
            frames: List["ImageObject"] = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())

            frames = self._regularize_images(frames, **kwargs)
            results.append(frames)

        return results

    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512 * 512),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_resolution=getattr(processor, "video_resolution", 128 * 128),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 64),
            )
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return messages

    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        r"""
        Pre-processes token ids after tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        r"""
        Builds batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos
        """
        self._validate_input(images, videos)
        return {}


class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        image_seqlen = getattr(processor, "image_seqlen") if self.expand_mm_tokens else 1
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class LlavaNextPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        if "image_sizes" in mm_inputs:
            image_sizes = iter(mm_inputs["image_sizes"])

        if "pixel_values" in mm_inputs:
            height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if getattr(processor, "vision_feature_select_strategy") == "default":
                        image_seqlen -= 1
                else:
                    image_seqlen = 1

                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class LlavaNextVideoPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        if "pixel_values" in mm_inputs:
            image_sizes = iter(mm_inputs["image_sizes"])
            height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))
            for message in messages:
                content = message["content"]
                while IMAGE_PLACEHOLDER in content:
                    if self.expand_mm_tokens:
                        orig_height, orig_width = next(image_sizes)
                        image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                        if getattr(processor, "vision_feature_select_strategy") == "default":
                            image_seqlen -= 1
                    else:
                        image_seqlen = 1

                    num_image_tokens += 1
                    content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

                message["content"] = content.replace("{{image}}", self.image_token)

        if "pixel_values_videos" in mm_inputs:
            pixel_values_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
            height, width = get_image_size(pixel_values_video[0])
            num_frames = pixel_values_video.shape[0]  # frame dim is always after batch dim
            image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
            video_seqlen = image_seqlen // 4 * num_frames  # divide by 4 needed for avg pooling layer
            video_seqlen = video_seqlen if self.expand_mm_tokens else 1
            for message in messages:
                content = message["content"]
                while VIDEO_PLACEHOLDER in content:
                    num_video_tokens += 1
                    content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)

                message["content"] = content.replace("{{video}}", self.video_token)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class PaliGemmaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)

            message["content"] = content.replace("{{image}}", "")

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        self._validate_input(images, videos)
        num_images = len(images)
        image_seqlen = num_images * getattr(processor, "image_seqlen") if self.expand_mm_tokens else 0  # skip mm token
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        input_ids = [image_token_id] * image_seqlen + input_ids
        if labels is not None:
            labels = [IGNORE_INDEX] * image_seqlen + labels

        return input_ids, labels

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        seqlens = [len(input_ids) for input_ids in batch_ids]
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


class PixtralPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        patch_size = getattr(processor, "patch_size")
        image_token = getattr(processor, "image_token")
        image_break_token = getattr(processor, "image_break_token")
        image_end_token = getattr(processor, "image_end_token")

        num_image_tokens = 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_input_sizes = mm_inputs.get("image_sizes", None)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if image_input_sizes is None:
                    raise ValueError("Cannot get image input sizes.")

                if self.expand_mm_tokens:
                    image_size = image_input_sizes[0][num_image_tokens]
                    height, width = image_size
                    num_height_tokens = height // patch_size
                    num_width_tokens = width // patch_size
                    replace_tokens = [[image_token] * num_width_tokens + [image_break_token]] * num_height_tokens
                    replace_tokens = [item for sublist in replace_tokens for item in sublist]  # flatten list
                    replace_tokens[-1] = image_end_token
                    replace_str = "".join(replace_tokens)
                else:
                    replace_str = image_token

                content = content.replace(IMAGE_PLACEHOLDER, replace_str, 1)
                num_image_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        if mm_inputs.get("pixel_values"):
            mm_inputs["pixel_values"] = mm_inputs["pixel_values"][0]

        mm_inputs.pop("image_sizes", None)
        return mm_inputs


class Qwen2vlPlugin(BasePlugin):
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @override
    def _regularize_videos(self, videos: Sequence["VideoInput"], **kwargs) -> List[List["ImageObject"]]:
        results = []
        for video in videos:
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            total_frames = video_stream.frames
            sample_frames = self._get_video_sample_frames(video_stream, **kwargs)
            sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
            frames: List["ImageObject"] = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())

            if len(frames) % 2 != 0:  # qwen2-vl requires even number of frames
                frames.append(frames[-1])

            frames = self._regularize_images(frames, **kwargs)
            results.append(frames)

        return results

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_PLACEHOLDER} tokens.")

                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_start|>{self.image_token * image_seqlen}<|vision_end|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"`len(videos)` is less than the number of {VIDEO_PLACEHOLDER} tokens.")

                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER, f"<|vision_start|>{self.video_token * video_seqlen}<|vision_end|>", 1
                )
                num_video_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
class Qwen2vlStreamPlugin(BasePlugin):
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @override
    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        r"""
        Computes video sample frames according to fps.
        """
        video_fps: float = kwargs.get("video_fps")
        video_maxlen: int = kwargs.get("video_maxlen")
        total_frames = video_stream.frames
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        # sample_frames = min(total_frames, video_maxlen, sample_frames)
        sample_frames = min(total_frames, sample_frames)
        return math.floor(sample_frames)

    @override
    def _regularize_videos(
            self,
            videos: Sequence["VideoInput"],
            video_time_segs: Sequence["List"],
            **kwargs) -> List[List["ImageObject"]]:

        results = []
        frame_times = []
        for video, time_seg in zip(videos, video_time_segs):
            t_start, t_end = time_seg
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            total_frames = video_stream.frames
            sample_frames = self._get_video_sample_frames(video_stream, **kwargs)
            sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
            sample_times = np.linspace(0, float(video_stream.duration * video_stream.time_base), sample_frames)

            sample_indices_seg, sample_times_seg = [], []
            for idx, t in zip(sample_indices, sample_times):
                if t_start <= t <= t_end:
                    sample_indices_seg.append(idx)
                    sample_times_seg.append(t)
            video_maxlen: int = kwargs.get("video_maxlen")
            if len(sample_indices_seg) > video_maxlen:
                sample_indices_seg = sample_indices_seg[-video_maxlen:]
                sample_times_seg = sample_times_seg[-video_maxlen:]

            frames: List["ImageObject"] = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices_seg:
                    frames.append(frame.to_image())

            if len(frames) % 2 != 0:  # qwen2-vl requires even number of frames
                frames.append(frames[-1])
                sample_times_seg.append(sample_times_seg[-1])

            frames = self._regularize_images(frames, **kwargs)
            results.append(frames)
            frame_times.append(sample_times_seg)

        return results, frame_times

    @override
    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
        video_time_segs: Sequence["List"],
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512 * 512),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            videos, frame_times = self._regularize_videos(
                videos,
                video_time_segs=video_time_segs,
                image_resolution=getattr(processor, "video_resolution", 128 * 128),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 64),
            )
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs['frame_times'] = [torch.FloatTensor(t[::2]) for t in frame_times]

        return mm_inputs

    '''
    接下来都是为了加速数据集的预处理，不然预处理太慢了
    '''
    def _get_fake_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
        video_time_segs: Sequence["List"],
    ):
        '''图片部分正常处理'''
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512 * 512),
            )
            input_dict["images"] = images
        mm_inputs = {}
        if input_dict.get("images") is not None:
            mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))

        # 视频部分简单处理
        def get_video_grid_thw():
            video_grid_thw = []
            # _regularize_videos() 中的过程
            video_frame_shapes = []
            frame_times = []
            for video, time_seg in zip(videos, video_time_segs):
                t_start, t_end = time_seg
                container = av.open(video, "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                frame_width, frame_height = video_stream.codec_context.width, video_stream.codec_context.height

                total_frames = video_stream.frames
                sample_frames = self._get_video_sample_frames(
                    video_stream,
                    video_fps=getattr(processor, "video_fps", 2.0),
                    video_maxlen=getattr(processor, "video_maxlen", 64),
                )
                sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
                sample_times = np.linspace(0, float(video_stream.duration * video_stream.time_base), sample_frames)
                sample_indices_seg, sample_times_seg = [], []
                for idx, t in zip(sample_indices, sample_times):
                    if t_start <= t <= t_end:
                        sample_indices_seg.append(idx)
                        sample_times_seg.append(t)
                video_maxlen = getattr(processor, "video_maxlen", 64)
                if len(sample_indices_seg) > video_maxlen:
                    sample_indices_seg = sample_indices_seg[-video_maxlen:]
                    sample_times_seg = sample_times_seg[-video_maxlen:]

                # 不需要真的读取视频
                # frames: List["ImageObject"] = []
                # container.seek(0)
                # for frame_idx, frame in enumerate(container.decode(video_stream)):
                #     if frame_idx in sample_indices_seg:
                #         frames.append(frame.to_image())

                if len(sample_times_seg) % 2 != 0:
                    sample_times_seg.append(sample_times_seg[-1])

                sample_frame_shapes = [(frame_width, frame_height)] * len(sample_times_seg)
                sample_frame_shapes = _regularize_images_shape(sample_frame_shapes, image_resolution=getattr(processor, "video_resolution", 128 * 128))

                # video_frame_shapes.append(sample_frame_shapes)
                # frame_times.append(sample_times_seg)

                # image_processor 过程
                new_width, new_height = sample_frame_shapes[0]
                grid_thw = _process_images_shape(len(sample_frame_shapes), new_width, new_height, image_processor)
                video_grid_thw.append(grid_thw)
                frame_times.append(sample_times_seg[::2])
            video_grid_thw = torch.tensor(video_grid_thw)
            return video_grid_thw, frame_times

        def _regularize_images_shape(image_shapes, image_resolution):
            output_shapes = []
            for width, height in image_shapes:
                if (width * height) > image_resolution:
                    resize_factor = math.sqrt(image_resolution / (width * height))
                    width, height = int(width * resize_factor), int(height * resize_factor)

                if min(width, height) < 28:
                    width, height = max(width, 28), max(height, 28)

                if width / height > 200:
                    width, height = height * 180, height

                if height / width > 200:
                    width, height = width, width * 180

                output_shapes.append((width, height))

            return output_shapes

        def _process_images_shape(num_frame, width, height, image_processor):
            if num_frame == 1:
                num_frame = 2
            if image_processor.do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=image_processor.patch_size * image_processor.merge_size,
                    min_pixels=image_processor.min_pixels,
                    max_pixels=image_processor.max_pixels,
                )
            else:
                resized_height, resized_width = height, width

            grid_t = num_frame // image_processor.temporal_patch_size
            grid_h, grid_w = resized_height // image_processor.patch_size, resized_width // image_processor.patch_size
            return grid_t, grid_h, grid_w

        video_grid_thw, frame_times = get_video_grid_thw()
        mm_inputs["video_grid_thw"] = video_grid_thw
        mm_inputs["frame_times"] = frame_times
        return mm_inputs


    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        # import pdb; pdb.set_trace()
        # print('Debug: 处理messages')

        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2

        # 判断视频应该怎么分段
        video_time_segs = []
        for message in messages:
            content = message["content"]
            if VIDEO_PLACEHOLDER in content:
                time = message['time']
                for i in range(0, len(time), 2):
                    video_time_segs.append([time[i], time[i + 1]])

        # import pdb; pdb.set_trace()
        # print("获取视频尺寸")

        # mm_inputs = self._get_mm_inputs(images, videos, processor, video_time_segs)
        mm_inputs = self._get_fake_mm_inputs(images, videos, processor, video_time_segs)

        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            content_stream = deepcopy(message['content'])
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_PLACEHOLDER} tokens.")

                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_start|>{self.image_token * image_seqlen}<|vision_end|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"`len(videos)` is less than the number of {VIDEO_PLACEHOLDER} tokens.")

                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER, f"<|vision_start|>{self.video_token * video_seqlen}<|vision_end|>", 1
                )

                # TODO: 使用更规范的实现方式
                # 用于判断哪些位置代表这一帧结束，用于进行stream_head的训练
                # 借用一下现有的special token:
                # <|video_pad|>: 普通的 video token
                # <|im_end|>: 每一帧的最后一个 video token
                # <|im_start|>: 回复前的最后一帧的最后1个 video token

                # import pdb; pdb.set_trace()
                # print('Debug: 设置content_stream')
                frame_num = video_grid_thw[num_video_tokens][0]
                frame_seqlen = video_grid_thw[num_video_tokens][1:].prod() // merge_length if self.expand_mm_tokens else 1
                video_content = ((self.video_token * (frame_seqlen - 1) + NO_RESPONSE_TOKEN) * (frame_num - 1) +
                                 (self.video_token * (frame_seqlen - 1) + DO_RESPONSE_TOKEN))
                content_stream = content_stream.replace(VIDEO_PLACEHOLDER, f"<|vision_start|>{video_content}<|vision_end|>", 1)

            message["content"] = content
            message["content_stream"] = content_stream

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
        video_time_segs: Sequence["List"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        # import pdb; pdb.set_trace()
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor, video_time_segs)






class Qwen2vlStreamPluginV2(BasePlugin):
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @override
    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        r"""
        Computes video sample frames according to fps.
        """
        video_fps: float = kwargs.get("video_fps")
        video_maxlen: int = kwargs.get("video_maxlen")
        total_frames = video_stream.frames
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        # sample_frames = min(total_frames, sample_frames)
        return math.floor(sample_frames)

    @override
    def _regularize_videos(
            self,
            videos: Sequence["VideoInput"],
            video_sample_idxs: Sequence["List"],
            **kwargs) -> List[List["ImageObject"]]:

        results = []
        for video, sample_indices_seg in zip(videos, video_sample_idxs):
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            frames: List["ImageObject"] = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices_seg:
                    frames.append(frame.to_image())
            frames = self._regularize_images(frames, **kwargs)
            results.append(frames)
        return results

    @override
    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
        video_sample_idxs: Sequence["List"],
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        # import pdb; pdb.set_trace()
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512 * 512),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                video_sample_idxs=video_sample_idxs,
                image_resolution=getattr(processor, "video_resolution", 128 * 128),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 64),
            )
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs

    '''
    这个函数是为了加速数据集的预处理，并不实际读取视频内容
    仅仅用于process_messages中， videos 和 video_time_segs 只能来自单条对话数据
    '''
    def _get_fake_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
        video_time_segs: Sequence["List"],
    ):
        # import pdb; pdb.set_trace()
        '''图片部分正常处理'''
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512 * 512),
            )
            input_dict["images"] = images
        mm_inputs = {}
        if input_dict.get("images") is not None:
            mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))

        # 视频部分简单处理
        # 先收集视频信息
        video_infos = {}
        for video in videos:
            if video in video_infos.keys():
                continue
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            frame_width, frame_height = video_stream.codec_context.width, video_stream.codec_context.height
            total_frames = video_stream.frames
            duration = float(video_stream.duration * video_stream.time_base)
            fps = float(total_frames) / duration
            video_infos[video] = {
                "width": frame_width,
                "height": frame_height,
                "duration": duration,
                "frame_num": total_frames,
                "fps": fps
            }

        # 先处理一下time_seg
        total_duration = 0
        for i in range(len(video_time_segs)):
            video = videos[i]
            video_duration = video_infos[video]["duration"]
            t_start, t_end = video_time_segs[i]
            t_start, t_end = max(t_start, 0), min(t_end, video_duration)
            total_duration += t_end - t_start
            video_time_segs[i] = [t_start, t_end]

        # 计算每段的shape和frame index
        def _regularize_images_shape(image_shapes, image_resolution):
            output_shapes = []
            for width, height in image_shapes:
                if (width * height) > image_resolution:
                    resize_factor = math.sqrt(image_resolution / (width * height))
                    width, height = int(width * resize_factor), int(height * resize_factor)

                if min(width, height) < 28:
                    width, height = max(width, 28), max(height, 28)

                if width / height > 200:
                    width, height = height * 180, height

                if height / width > 200:
                    width, height = width, width * 180

                output_shapes.append((width, height))

            return output_shapes

        def _process_images_shape(num_frame, width, height, image_processor):
            if num_frame == 1:
                num_frame = 2
            if image_processor.do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=image_processor.patch_size * image_processor.merge_size,
                    min_pixels=image_processor.min_pixels,
                    max_pixels=image_processor.max_pixels,
                )
            else:
                resized_height, resized_width = height, width

            grid_t = num_frame // image_processor.temporal_patch_size
            grid_h, grid_w = resized_height // image_processor.patch_size, resized_width // image_processor.patch_size
            return grid_t, grid_h, grid_w


        # _regularize_videos() 中的过程
        video_fps = getattr(processor, "video_fps", 2.0)
        video_maxlen = getattr(processor, "video_maxlen", 64)
        video_grid_thw = []

        # 给每段分配帧数

        frame_nums = []
        for video, time_seg in zip(videos, video_time_segs):
            video_info = video_infos[video]
            frame_width, frame_height = video_info["width"], video_info["height"]
            real_fps = video_info["fps"]
            video_duration = video_info["duration"]
            total_frames = video_info['frame_num']

            # 先计算这一段需要采样多少帧
            t_start, t_end = time_seg
            seg_duration = t_end - t_start
            frame_num = min(seg_duration * video_fps, seg_duration * real_fps)
            frame_num = min(frame_num, video_maxlen * seg_duration / total_duration)
            frame_num = math.floor(frame_num)
            frame_num = max(frame_num, 2)   # 最少采集2帧
            if frame_num % 2 != 0:
                # 必须是偶数
                frame_num -= 1
            frame_nums.append(frame_num)

        # 此时各段的采样帧数可能加起来超过 video_maxlen
        current_total = sum(frame_nums)
        # 如果超过，则对各段进行迭代调整，每次从那些帧数大于2的段减少2帧，直到总数不超过总数要求
        while current_total > video_maxlen:
            # import pdb; pdb.set_trace()
            # print('DEBUG: frame index!')

            reduced = False
            for i in range(len(frame_nums)):
                if frame_nums[i] > 2:
                    frame_nums[i] -= 2
                    current_total -= 2
                    reduced = True
                    if current_total <= video_maxlen:
                        break
            if not reduced:
                # 如果所有段都已经是2帧，无法再减少，则退出循环
                break

        # 确定采样的frame idx
        frame_times, frame_idxs = [], []
        for video, time_seg, frame_num in zip(videos, video_time_segs, frame_nums):
            video_info = video_infos[video]
            frame_width, frame_height = video_info["width"], video_info["height"]
            real_fps = video_info["fps"]
            video_duration = video_info["duration"]
            total_frames = video_info['frame_num']

            t_start, t_end = time_seg
            sample_times = np.linspace(t_start, t_end, frame_num, endpoint=False)
            sample_idxs = (sample_times * real_fps).round().astype(np.int32)
            sample_idxs = sample_idxs.clip(min=0, max=total_frames-1)

            sample_frame_shapes = [(frame_width, frame_height)] * len(sample_idxs)
            sample_frame_shapes = _regularize_images_shape(sample_frame_shapes, image_resolution=getattr(processor, "video_resolution", 128 * 128))

            # image_processor 过程
            new_width, new_height = sample_frame_shapes[0]
            grid_thw = _process_images_shape(len(sample_frame_shapes), new_width, new_height, image_processor)
            video_grid_thw.append(torch.tensor(grid_thw))

            frame_idxs.append(sample_idxs)
            frame_times.append(sample_times[::2])

        mm_inputs["video_grid_thw"] = video_grid_thw
        mm_inputs["frame_times"] = frame_times
        mm_inputs["frame_idxs"] = frame_idxs
        return mm_inputs


    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        # import pdb; pdb.set_trace()
        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2

        # 判断视频应该怎么分段
        video_time_segs = []
        for message in messages:
            content = message["content"]
            if VIDEO_PLACEHOLDER in content:
                time = message['time']
                for i in range(0, len(time), 2):
                    video_time_segs.append([time[i], time[i + 1]])

        # import pdb; pdb.set_trace()
        # print("获取视频尺寸")

        # mm_inputs = self._get_mm_inputs(images, videos, processor, video_time_segs)
        mm_inputs = self._get_fake_mm_inputs(images, videos, processor, video_time_segs)

        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        # 必须先判断是不是需要进行视频的拼接
        # import pdb; pdb.set_trace()
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while "<video><+><video>" in content:
                # 2段视频拼接到一起
                assert (video_grid_thw[num_video_tokens][1:] == video_grid_thw[num_video_tokens + 1][1:]).all()
                video_grid_thw[num_video_tokens][0] += video_grid_thw[num_video_tokens + 1][0]
                del video_grid_thw[num_video_tokens + 1]
                content = content.replace("<video><+><video>", "<video>", 1)
            message["content"] = content

            while "<video>" in content:
                content = content.replace("<video>", "", 1)
                num_video_tokens += 1

        # 正常插入占位token
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            content_stream = deepcopy(message['content'])
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_PLACEHOLDER} tokens.")

                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_start|>{self.image_token * image_seqlen}<|vision_end|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"`len(videos)` is less than the number of {VIDEO_PLACEHOLDER} tokens.")

                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER, f"<|vision_start|>{self.video_token * video_seqlen}<|vision_end|>", 1
                )

                # TODO: 使用更规范的实现方式
                # 用于判断哪些位置代表这一帧结束，用于进行stream_head的训练
                # import pdb; pdb.set_trace()
                # print('Debug: 设置content_stream')
                frame_num = video_grid_thw[num_video_tokens][0]
                frame_seqlen = video_grid_thw[num_video_tokens][1:].prod() // merge_length if self.expand_mm_tokens else 1
                video_content = (self.video_token * (frame_seqlen - 1) + FRAME_END_TOKEN) * frame_num
                content_stream = content_stream.replace(VIDEO_PLACEHOLDER, f"<|vision_start|>{video_content}<|vision_end|>", 1)
                num_video_tokens += 1

            message["content"] = content
            message["content_stream"] = content_stream

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(video_grid_thw) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        # 必须在这个过程中把video sample index计算清楚, 也作为返回
        # 因为涉及到多段视频拼接成完整的一段，所以video_grid_thw也必须返回
        frame_idxs = mm_inputs.get("frame_idxs", [])
        frame_times = mm_inputs.get("frame_times", [])

        return messages, frame_idxs, frame_times, video_grid_thw


    '''
    这个函数用于collator中，需要实际读取视频内容
    '''
    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
        video_sample_idxs: Sequence["List"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor, video_sample_idxs)





class Qwen2vlStreamPluginV3(Qwen2vlStreamPluginV2):

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        # import pdb; pdb.set_trace()
        # print("DebugV3: process_messages")

        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2

        # 判断视频应该怎么分段
        video_time_segs = []
        for message in messages:
            content = message["content"]
            if VIDEO_PLACEHOLDER in content:
                time = message['time']
                for i in range(0, len(time), 2):
                    video_time_segs.append([time[i], time[i + 1]])

        # import pdb; pdb.set_trace()
        # print("获取视频尺寸")

        # mm_inputs = self._get_mm_inputs(images, videos, processor, video_time_segs)
        mm_inputs = self._get_fake_mm_inputs(images, videos, processor, video_time_segs)

        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        # 必须先判断是不是需要进行视频的拼接
        # import pdb; pdb.set_trace()
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while "<video><+><video>" in content:
                # 2段视频拼接到一起
                assert (video_grid_thw[num_video_tokens][1:] == video_grid_thw[num_video_tokens + 1][1:]).all()
                video_grid_thw[num_video_tokens][0] += video_grid_thw[num_video_tokens + 1][0]
                del video_grid_thw[num_video_tokens + 1]
                content = content.replace("<video><+><video>", "<video>", 1)
            message["content"] = content

            while "<video>" in content:
                content = content.replace("<video>", "", 1)
                num_video_tokens += 1

        # 正常插入占位token
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            content_stream = deepcopy(message['content'])
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_PLACEHOLDER} tokens.")

                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_start|>{self.image_token * image_seqlen}<|vision_end|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"`len(videos)` is less than the number of {VIDEO_PLACEHOLDER} tokens.")
                # import pdb; pdb.set_trace()
                # print('Debug: 设置content_stream')

                # 每一帧最后都需要加上<|vision_end|><|im_end|>
                frame_num = video_grid_thw[num_video_tokens][0]
                frame_seqlen = video_grid_thw[num_video_tokens][1:].prod() // merge_length if self.expand_mm_tokens else 1
                video_content = ("<|vision_start|>" +
                                 (self.video_token * frame_seqlen + '<|vision_end|><|im_end|>') * (frame_num -1) +
                                 (self.video_token * frame_seqlen + '<|vision_end|>'))

                mask_content = ("<|vision_start|>" +
                                 (self.video_token * frame_seqlen + FRAME_PAD_TOKEN * 2) * (frame_num -1) +
                                 (self.video_token * frame_seqlen + '<|vision_end|>'))

                content = content.replace(VIDEO_PLACEHOLDER, video_content, 1)
                content_stream = content_stream.replace(VIDEO_PLACEHOLDER, mask_content, 1)
                num_video_tokens += 1
            message["content"] = content
            message["content_stream"] = content_stream

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(video_grid_thw) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        # 必须在这个过程中把video sample index计算清楚, 也作为返回
        # 因为涉及到多段视频拼接成完整的一段，所以video_grid_thw也必须返回
        frame_idxs = mm_inputs.get("frame_idxs", [])
        frame_times = mm_inputs.get("frame_times", [])

        return messages, frame_idxs, frame_times, video_grid_thw



class VideoLlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        num_frames = 0
        has_images = "pixel_values_images" in mm_inputs
        has_videos = "pixel_values_videos" in mm_inputs
        if has_images or has_videos:
            if self.expand_mm_tokens:
                if has_images:
                    height, width = get_image_size(to_numpy_array(mm_inputs.get("pixel_values_images")[0]))
                    num_frames = 1

                if has_videos:
                    pixel_values_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
                    height, width = get_image_size(pixel_values_video[0])
                    num_frames = pixel_values_video.shape[0]  # frame dim is always after batch dim

                image_seqlen = (height // processor.patch_size) * (width // processor.patch_size) + 1
                video_seqlen = image_seqlen * num_frames
                if getattr(processor, "vision_feature_select_strategy") == "default":
                    image_seqlen -= 1
            else:
                image_seqlen, video_seqlen = 1, 1

            for message in messages:
                content = message["content"]
                while IMAGE_PLACEHOLDER in content:
                    num_image_tokens += 1
                    content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

                while VIDEO_PLACEHOLDER in content:
                    num_video_tokens += 1
                    content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)

                content = content.replace("{{image}}", self.image_token)
                message["content"] = content.replace("{{video}}", self.video_token)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class MllamaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            num_image_tokens += content.count(IMAGE_PLACEHOLDER)
            message["content"] = content.replace(IMAGE_PLACEHOLDER, self.image_token)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs for mllama because its image processor only accepts List[List[ImageInput]].

        Returns:
            pixel_values: tensor with shape
                          (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
                          For example, (2, 1, 4, 3, 560, 560).
            aspect_ratio_ids: tensor with shape (batch_size, max_num_images). For example, (2, 1).
            aspect_ratio_mask: tensor with shape (batch_size, max_num_images, max_image_tiles). For example, (2, 1, 4).
            num_tiles: List[List[int]] with shape (batch_size, num_images_in_batch). For example, (2, 1).
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        images = self._regularize_images(images, image_resolution=getattr(processor, "image_resolution", 512 * 512))
        return image_processor([[image] for image in images], return_tensors="pt")

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        if len(images) != len(batch_ids):
            raise ValueError("Mllama only supports one image per sample.")

        mm_inputs = self._get_mm_inputs(images, videos, processor)
        num_tiles = mm_inputs.pop("num_tiles")
        image_token_id = getattr(processor, "image_token_id")
        max_image_tiles = getattr(processor.image_processor, "max_image_tiles")
        cross_attention_token_mask = [
            get_cross_attention_token_mask(input_ids, image_token_id) for input_ids in batch_ids
        ]
        mm_inputs["cross_attention_mask"] = torch.from_numpy(
            convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=num_tiles,
                max_num_tiles=max_image_tiles,
                length=max(len(input_ids) for input_ids in batch_ids),
            )
        )  # shape: (batch_size, length, max_num_images, max_num_tiles)
        return mm_inputs


PLUGINS = {
    "base": BasePlugin,
    "llava": LlavaPlugin,
    "llava_next": LlavaNextPlugin,
    "llava_next_video": LlavaNextVideoPlugin,
    "paligemma": PaliGemmaPlugin,
    "pixtral": PixtralPlugin,
    "qwen2_vl": Qwen2vlPlugin,
    "qwen2_vl_stream": Qwen2vlStreamPlugin,
    "qwen2_vl_stream_v2": Qwen2vlStreamPluginV2,
    "qwen2_vl_stream_v3": Qwen2vlStreamPluginV3,
    "video_llava": VideoLlavaPlugin,
    "mllama": MllamaPlugin,
}


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
) -> "BasePlugin":
    plugin_class = PLUGINS.get(name, None)
    if plugin_class is None:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return plugin_class(image_token, video_token)
