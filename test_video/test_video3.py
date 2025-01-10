from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info, fetch_video
import torch
from src.llamafactory.monkey_patch.qwen2_vl_monkey_patch import Qwen2VLStream
from tqdm import tqdm


model_ckpt = '/afs/zengwang/projects/task_define_service/LLaMA-Factory/work_dirs/stream_head_only_1/Stream-Qwen2-VL-7B-Instruct'
model = Qwen2VLStream.from_pretrained(
    model_ckpt,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
model = model.eval()

# default processer
processor = AutoProcessor.from_pretrained(model_ckpt)


video_path = '/afs/zengwang/projects/task_define_service/data/video_event/push-up_2.mp4'
query = 'Please narrate the video in real time.'
fps = 2

# video_path = '/afs/zengwang/projects/task_define_service/data/shot2story-videos_release_134k/W26nTWGbf3g.8.mp4'
# query = 'Please concisely narrate the video in real time.'
# fps = 4

video_info = {
    "type": "video",
    "video": video_path,
    "max_pixels": 256 * 256,
    "fps": fps,
}
all_frames = fetch_video(video_info)

video_token_id = 151656

text_historys = [{"role": 'user', 'content': query}]
for idx_frame in tqdm(range(2, len(all_frames), 2)):
    video_message = {"role": 'user', 'content': [video_info, {"type": "text", "text": ''}]}
    messages = text_historys + [video_message]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    video_inputs = [all_frames[:idx_frame]]
    inputs = processor(text=[text],images=None, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    with torch.no_grad():
        output = model.forward(**inputs)
    last_frame_token_index = (inputs['input_ids'] == video_token_id).nonzero(as_tuple=True)[1].max().item()
    stream_logits = output.stream_logits
    last_logits = stream_logits[0, last_frame_token_index]
    if last_logits[1] > last_logits[0]:
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0]
        print(f'At frame {idx_frame}:\nAssistant: {output_text}')
        text_historys.append({"role": 'assistant', 'content': output_text})