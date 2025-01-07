cd /afs/zengwang/projects/task_define_service/LLaMA-Factory

python src/train.py \
  --stage sft \
  --do_train \
  --model_name_or_path /afs/zengwang/ckpt/Stream-Qwen2-VL-7B-Instruct \
  --template qwen2_vl_stream \
  --overwrite_cache \
  --overwrite_output_dir \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --mask_history \
  --freeze_vision_tower \
  --finetuning_type freeze \
  --freeze_trainable_layers 0 \
  --freeze_extra_modules stream_head \
  --video_resolution 147456 \
  --video_fps 2 \
  --video_maxlen 64 \
  --cutoff_len 8192 \
  --dataset MMDuetIT_dvc_stream,MMDuetIT_magqa_stream  \
  --image_dir /afs/zengwang/projects/task_define_service/data/shot2story-videos_release_134k \
  --num_train_epochs 1 \
  --output_dir work_dirs/stream_head_only_1




~/ads-cli sync \
/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed \
s3://196FFD00B6184227B65B3D92C01A8724:DD1D004D80834448B276F125F8310F2A@zengwang.aoss.cn-sh-01.sensecoreapi-oss.cn/data/processed/MMDuetIT


/afs/zengwang/ads-cli sync \
s3://196FFD00B6184227B65B3D92C01A8724:DD1D004D80834448B276F125F8310F2A@zengwang.aoss-internal.cn-sh-01.sensecoreapi-oss.cn/data/processed/MMDuetIT \
/afs/zengwang/projects/task_define_service/data/processed/MMDuetIT

