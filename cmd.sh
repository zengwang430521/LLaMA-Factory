rm -rf ~/.ssh
ln -s /afs/zengwang/.ssh ~/.ssh

cd /afs/zengwang/projects/task_define_service/LLaMA-Factory
export PYTHONPATH=$PYTHONPATH:/afs/zengwang/projects/task_define_service/LLaMA-Factory/src


python src/train.py \
  --stage sft \
  --do_train \
  --model_name_or_path /afs/zengwang/ckpt/Stream-Qwen2-VL-7B-Instruct \
  --template qwen2_vl_stream \
  --overwrite_cache \
  --overwrite_output_dir \
  --bf16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --mask_history \
  --freeze_vision_tower \
  --finetuning_type lora \
  --lora_rank 8 \
  --video_resolution 65536 \
  --video_fps 2 \
  --video_maxlen 64 \
  --cutoff_len 4096 \
  --num_train_epochs 1 \
  --output_dir work_dirs/debug \
  --dataset MMDuetIT_dvc_stream_sample100 \
  --image_dir /afs/zengwang/projects/task_define_service/data/shot2story-videos_release_134k


  --dataset mllm_video_stream_demo




export PYTHONPATH=$PYTHONPATH:/afs/zengwang/projects/task_define_service/LLaMA-Factory/src

python test_video/test_video3.py \
  --model_path /afs/zengwang/projects/task_define_service/LLaMA-Factory/work_dirs/stream_head_only_1/Stream-Qwen2-VL-7B-Instruct \
  --video_path /afs/zengwang/projects/task_define_service/data/video_event/push-up_2.mp4


python test_video/test_video3.py \
  --model_path /afs/zengwang/ckpt/Stream-Qwen2-VL-7B-Instruct \
  --lora_path /afs/zengwang/projects/task_define_service/LLaMA-Factory/work_dirs/stream_lora_1/checkpoint-500 \
  --video_path /afs/zengwang/projects/task_define_service/data/video_event/push-up_2.mp4




  --video_path /afs/zengwang/projects/task_define_service/data/video_event/push-up_2.mp4
  --video_path /afs/zengwang/projects/task_define_service/data/shot2story-videos_release_134k/W26nTWGbf3g.8.mp4





~/ads-cli sync \
/home/SENSETIME/zengwang/myprojects/task_define_service/data/MMDuetIT/shot2story/annotations/processed \
s3://196FFD00B6184227B65B3D92C01A8724:DD1D004D80834448B276F125F8310F2A@zengwang.aoss.cn-sh-01.sensecoreapi-oss.cn/data/processed/MMDuetIT


/afs/zengwang/ads-cli sync \
s3://196FFD00B6184227B65B3D92C01A8724:DD1D004D80834448B276F125F8310F2A@zengwang.aoss-internal.cn-sh-01.sensecoreapi-oss.cn/data/processed/MMDuetIT \
/afs/zengwang/projects/task_define_service/data/processed/MMDuetIT

