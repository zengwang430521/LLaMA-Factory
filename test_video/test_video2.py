from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info, fetch_video
import torch


model_ckpt = '/afs/zengwang/ckpt/Qwen2-VL-7B-Instruct'
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_ckpt,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(model_ckpt)


# video_path = '/afs/zengwang/projects/task_define_service/data/video_event/push-up_2.mp4'
# time_spots = [2, 15, 21, 43, 54]
# query = 'Please narrate the video in real time.'
# fps = 4

video_path = '/afs/zengwang/projects/task_define_service/data/shot2story-videos_release_134k/W26nTWGbf3g.8.mp4'
time_spots = [2, 8, 14, 20]
query = 'Please concisely narrate the video in real time.'
fps = 4

video_info = {
    "type": "video",
    "video": video_path,
    "max_pixels": 256 * 28 * 28,
    "fps": fps,
}
all_frames = fetch_video(video_info)

video_clips = []
idx_start = 0
for t in time_spots:
    idx_end = min(int(t * fps), all_frames.shape[0])
    video_clips.append(all_frames[idx_start:idx_end])
    idx_start = idx_end

# 初始messages
messages = [{
    "role": 'user',
    'content': [
        video_info,
        {"type": "text", "text": query},
    ]
}]
image_inputs = None
video_inputs = []
# import pdb; pdb.set_trace()
for t, clip in zip(time_spots, video_clips):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    video_inputs.append(clip)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output_text = output_text[0]
    print(f'Time {t} s:')
    print(F'Assistant: {output_text}')

    messages.append({"role": 'assistant', 'content': output_text})
    messages.append({"role": 'user', 'content': [video_info]})
