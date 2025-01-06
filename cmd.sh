cd /afs/zengwang/projects/task_define_service/LLaMA-Factory
python src/train.py \
  --stage sft \
  --do_train \
  --model_name_or_path /afs/zengwang/ckpt/Qwen2-VL-7B-Instruct \
  --dataset mllm_video_stream_demo \
  --template qwen2_vl_stream \
  --output_dir work_dirs/debug \
  --overwrite_cache \
  --overwrite_output_dir \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --mask_history \
  --new_special_tokens <|frame_end|>
