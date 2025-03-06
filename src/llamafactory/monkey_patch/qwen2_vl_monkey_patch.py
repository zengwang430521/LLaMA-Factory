from transformers.models.qwen2_vl.modeling_qwen2_vl import *
from transformers.models import *
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.models.auto import AutoModelForVision2Seq, AutoConfig
from torch.nn import BCELoss

_CONFIG_FOR_DOC = "Qwen2VLConfig"


class Qwen2VLStreamConfig(Qwen2VLConfig):
    model_type = "qwen2_vl_stream"

    def __init__(self, stream_loss_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.stream_loss_factor = stream_loss_factor


@dataclass
class Qwen2VLStreamOutput(ModelOutput):
    """
    Base class for Qwen2VL causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    stream_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen2VLStream(Qwen2VLForConditionalGeneration):
    config_class = Qwen2VLStreamConfig

    def __init__(self, config):
        super().__init__(config)
        self.stream_head = nn.Linear(config.hidden_size, 2, bias=False)     # 二分类，回复/不回复
        self.stream_loss_factor = config.stream_loss_factor
        self.post_init()

    @add_start_docstrings_to_model_forward(QWEN2_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        stream_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2VLStreamOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        # import pdb; pdb.set_trace()
        # print('Debug: 模型forward')
        # if not hasattr(self, "tokenizer"):
        #     from transformers.models.auto import AutoTokenizer
        #     self.tokenizer = AutoTokenizer.from_pretrained("/afs/zengwang/ckpt/Stream-Qwen2-VL-7B-Instruct")
        #     input_texts = self.tokenizer.batch_decode(input_ids)
        #     self.tokenizer.batch_decode(input_ids[labels!=-100])
        #     indexs = (stream_labels == 0).nonzero()
        #     for index in indexs:
        #         idx_batch, idx_token = index;
        #         print('-'* 50);
        #         print(index);
        #         idx_end = min(idx_token+20, input_ids.shape[1]-1);
        #         idx_begin = max(idx_token-20, 0);
        #         print(self.tokenizer.decode(input_ids[idx_batch, idx_begin:idx_token]) +
        #               '\n@\n' + self.tokenizer.decode(input_ids[idx_batch, idx_token]) + '\n@\n' +
        #               self.tokenizer.decode(input_ids[idx_batch, idx_token+1:idx_end]))
        #
        #         # print(self.tokenizer.decode(input_ids[idx_batch, idx_begin:idx_token]))
        #         # print(self.tokenizer.decode(input_ids[idx_batch, idx_token:idx_end]))
        #         (stream_labels == 0).nonzero()
        #         self.tokenizer.batch_decode(input_ids[stream_labels!=-100])

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # print('Debug')
        # seq_len = attention_mask.shape[1]
        # casual_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None and logits.requires_grad:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # stream head
        hidden_states = hidden_states.to(self.stream_head.weight.dtype)
        stream_logits = self.stream_head(hidden_states)
        if stream_labels is not None:
            # stream label 不需要做shift
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            stream_logits = stream_logits.float()
            loss_fct_stream = CrossEntropyLoss()
            # Flatten the tokens
            stream_logits = stream_logits.view(-1, 2)
            stream_labels = stream_labels.view(-1)
            # Enable model parallelism
            stream_labels = stream_labels.to(stream_logits.device)
            stream_loss = loss_fct_stream(stream_logits, stream_labels)
            if loss is not None:
                loss += stream_loss * self.stream_loss_factor
            else:
                loss = stream_loss * self.stream_loss_factor

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLStreamOutput(
            loss=loss,
            logits=logits,
            stream_logits=stream_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )


    # '''没有修改过'''
    # def get_rope_index(
    #     self,
    #     input_ids: torch.LongTensor,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
    #
    #     Explanation:
    #         Each embedding sequence contains vision embedding and text embedding or just contains text embedding.
    #
    #         For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
    #         Examples:
    #             input_ids: [T T T T T], here T is for text.
    #             temporal position_ids: [0, 1, 2, 3, 4]
    #             height position_ids: [0, 1, 2, 3, 4]
    #             width position_ids: [0, 1, 2, 3, 4]
    #
    #         For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
    #         and 1D rotary position embeddin for text part.
    #         Examples:
    #             Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
    #             input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
    #             vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    #             vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    #             vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    #             text temporal position_ids: [3, 4, 5, 6, 7]
    #             text height position_ids: [3, 4, 5, 6, 7]
    #             text width position_ids: [3, 4, 5, 6, 7]
    #             Here we calculate the text start position_ids as the max vision position_ids plus 1.
    #
    #     Args:
    #         input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
    #             Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
    #             it.
    #         image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
    #             The temporal, height and width of feature shape of each image in LLM.
    #         video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
    #             The temporal, height and width of feature shape of each video in LLM.
    #         attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
    #             Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    #
    #             - 1 for tokens that are **not masked**,
    #             - 0 for tokens that are **masked**.
    #
    #     Returns:
    #         position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
    #         mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    #     """
    #     import pdb; pdb.set_trace()
    #     spatial_merge_size = self.config.vision_config.spatial_merge_size
    #     image_token_id = self.config.image_token_id
    #     video_token_id = self.config.video_token_id
    #     vision_start_token_id = self.config.vision_start_token_id
    #     mrope_position_deltas = []
    #     if image_grid_thw is not None or video_grid_thw is not None:
    #         total_input_ids = input_ids
    #         position_ids = torch.ones(
    #             3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
    #         )
    #         image_index, video_index = 0, 0
    #         for i, input_ids in enumerate(total_input_ids):
    #             if attention_mask is not None:
    #                 input_ids = input_ids[attention_mask[i] == 1]
    #             image_nums, video_nums = 0, 0
    #             vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
    #             vision_tokens = input_ids[vision_start_indices + 1]
    #             image_nums = (vision_tokens == image_token_id).sum()
    #             video_nums = (vision_tokens == video_token_id).sum()
    #             input_tokens = input_ids.tolist()
    #             llm_pos_ids_list: list = []
    #             st = 0
    #             remain_images, remain_videos = image_nums, video_nums
    #             for _ in range(image_nums + video_nums):
    #                 if image_token_id in input_tokens and remain_images > 0:
    #                     ed_image = input_tokens.index(image_token_id, st)
    #                 else:
    #                     ed_image = len(input_tokens) + 1
    #                 if video_token_id in input_tokens and remain_videos > 0:
    #                     ed_video = input_tokens.index(video_token_id, st)
    #                 else:
    #                     ed_video = len(input_tokens) + 1
    #                 if ed_image < ed_video:
    #                     t, h, w = (
    #                         image_grid_thw[image_index][0],
    #                         image_grid_thw[image_index][1],
    #                         image_grid_thw[image_index][2],
    #                     )
    #                     image_index += 1
    #                     remain_images -= 1
    #                     ed = ed_image
    #                 else:
    #                     t, h, w = (
    #                         video_grid_thw[video_index][0],
    #                         video_grid_thw[video_index][1],
    #                         video_grid_thw[video_index][2],
    #                     )
    #                     video_index += 1
    #                     remain_videos -= 1
    #                     ed = ed_video
    #                 llm_grid_t, llm_grid_h, llm_grid_w = (
    #                     t.item(),
    #                     h.item() // spatial_merge_size,
    #                     w.item() // spatial_merge_size,
    #                 )
    #                 text_len = ed - st
    #
    #                 st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    #                 llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
    #
    #                 t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
    #                 h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
    #                 w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
    #                 llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
    #                 st = ed + llm_grid_t * llm_grid_h * llm_grid_w
    #
    #             if st < len(input_tokens):
    #                 st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    #                 text_len = len(input_tokens) - st
    #                 llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
    #
    #             llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    #             position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
    #             mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
    #         mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    #         return position_ids, mrope_position_deltas
    #     else:
    #         if attention_mask is not None:
    #             position_ids = attention_mask.long().cumsum(-1) - 1
    #             position_ids.masked_fill_(attention_mask == 0, 1)
    #             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
    #             max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
    #             mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    #         else:
    #             position_ids = (
    #                 torch.arange(input_ids.shape[1], device=input_ids.device)
    #                 .view(1, 1, -1)
    #                 .expand(3, input_ids.shape[0], -1)
    #             )
    #             mrope_position_deltas = torch.zeros(
    #                 [input_ids.shape[0], 1],
    #                 device=input_ids.device,
    #                 dtype=input_ids.dtype,
    #             )
    #
    #         return position_ids, mrope_position_deltas


AutoConfig.register('qwen2_vl_stream', Qwen2VLStreamConfig)
# AutoConfig.register('qwen2_vl_stream_v2', Qwen2VLStreamConfig)
AutoModelForVision2Seq.register(Qwen2VLStreamConfig, Qwen2VLStream)






class Qwen2VLStreamConfigV3(Qwen2VLConfig):
    model_type = "qwen2_vl_stream"

    def __init__(self, stream_loss_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.stream_loss_factor = stream_loss_factor


def fill_missing_pos_batch(pos, mask):
    """
    pos: torch.LongTensor, shape = (3, batch, n)
    mask: torch.BoolTensor, shape = (batch, n)

    对于 mask 为 False 的位置，所有行都使用 pos 第一行计算出的值进行填充，
    填充值为：pos[0, batch, 左侧最近有效索引] + (当前位置索引 - 左侧最近有效索引)。
    """
    batch, n = mask.shape  # batch 和 n 的大小
    # 生成索引 (batch, n)
    idx = torch.arange(n, device=pos.device).unsqueeze(0).expand(batch, n)

    # 对于 mask=False 的位置，将索引置为 -1
    valid_idx = torch.where(mask, idx, torch.full_like(idx, -1))

    # 计算沿 n 维的累计最大值，得到每个位置左侧最近的有效索引
    cum_max, _ = torch.cummax(valid_idx, dim=1)

    # 计算当前位置与左侧最近有效位置之间的差值
    diff = idx - cum_max

    # 从 pos 第一行中取出有效值，并加上 diff 得到填充值
    # pos[0] 的 shape 为 (batch, n)
    filled = pos[0].gather(dim=1, index=cum_max) + diff

    # 扩展 mask 到 shape (3, batch, n)
    mask_expanded = mask.unsqueeze(0).expand_as(pos)
    # 扩展 filled 到 shape (3, batch, n)
    filled_expanded = filled.unsqueeze(0).expand_as(pos)

    # mask 为 True 的位置保留原 pos 值，否则用 filled 替换
    result = torch.where(mask_expanded, pos, filled_expanded)
    return result


class Qwen2VLStreamV3(Qwen2VLForConditionalGeneration):
    config_class = Qwen2VLStreamConfigV3

    def __init__(self, config):
        super().__init__(config)
        self.stream_head = nn.Linear(config.hidden_size, 2, bias=False)     # 二分类，不回复/回复
        self.stream_loss_factor = config.stream_loss_factor
        self.post_init()

    @add_start_docstrings_to_model_forward(QWEN2_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        stream_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2VLStreamOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """

        import pdb; pdb.set_trace()
        print('Debug: 模型forward')

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)


        # 由于部分被mask的token需要参与stream_head的优化，所以他们需要有attn
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        seq_len = causal_mask.shape[-1]
        mask_with_indices = torch.cumsum(attention_mask, dim=1)
        expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
        attention_mask_4d_pad = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2))
        causal_mask = torch.where(attention_mask_4d_pad, torch.tensor(0, dtype=causal_mask.dtype), causal_mask)


        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            # attention_mask=attention_mask,
            attention_mask=causal_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None and logits.requires_grad:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # stream head
        hidden_states = hidden_states.to(self.stream_head.weight.dtype)
        stream_logits = self.stream_head(hidden_states)
        if stream_labels is not None:
            # stream label 不需要做shift
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            stream_logits = stream_logits.float()
            loss_fct_stream = CrossEntropyLoss()
            # Flatten the tokens
            stream_logits = stream_logits.view(-1, 2)
            stream_labels = stream_labels.view(-1)
            # Enable model parallelism
            stream_labels = stream_labels.to(stream_logits.device)
            stream_loss = loss_fct_stream(stream_logits, stream_labels)
            if loss is not None:
                loss += stream_loss * self.stream_loss_factor
            else:
                loss = stream_loss * self.stream_loss_factor

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLStreamOutput(
            loss=loss,
            logits=logits,
            stream_logits=stream_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )




    '''被mask的token也需要有position_ids'''
    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            # 被mask的部分也需要设置 position_ids
            import pdb; pdb.set_trace()
            position_ids = fill_missing_pos_batch(position_ids, mask=attention_mask)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


AutoConfig.register('qwen2_vl_stream_v3', Qwen2VLStreamConfigV3)
AutoModelForVision2Seq.register(Qwen2VLStreamConfigV3, Qwen2VLStreamV3)