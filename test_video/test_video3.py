from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info, fetch_video
import torch
from llamafactory.monkey_patch.qwen2_vl_monkey_patch import Qwen2VLStream
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_path", type=str, help="model_path")
    parser.add_argument("--video_path", type=str, help="video_path")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--query", type=str, default='Please narrate the video in real time.')


    args = parser.parse_args()
    model_ckpt = args.model_path
    video_path = args.video_path
    fps = args.fps
    query = args.query

    model = Qwen2VLStream.from_pretrained(
        model_ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model = model.eval()

    # default processer
    processor = AutoProcessor.from_pretrained(model_ckpt)




    video_info = {
        "type": "video",
        "video": video_path,
        "max_pixels": 256 * 256,
        "fps": fps,
    }
    all_frames = fetch_video(video_info)

    video_token_id = 151656

    text_historys = [{"role": 'user', 'content': query}]
    for idx_frame in range(2, len(all_frames), 2):
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
            # import pdb; pdb.set_trace()
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            video_inputs = [all_frames[:idx_frame]]
            inputs = processor(text=[text], images=None, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0]
        print(f'At time {idx_frame//2} s:\nAssistant: {output_text}\n')
        text_historys.append({"role": 'assistant', 'content': output_text})