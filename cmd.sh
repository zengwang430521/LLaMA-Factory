python src/train.py \
  --stage sft \
  --do_train \
  --model_name_or_path /afs/zengwang/ckpt/Qwen2-VL-7B-Instruct \
  --dataset mllm_demo \
  --template qwen2_vl \
  --output_dir work_dirs/debug \
  --overwrite_cache \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --weight_decay 0.1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --ddp_timeout 9000 \
  --learning_rate 5e-6 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --cutoff_len 4096 \
  --save_steps 1000 \
  --plot_loss \
  --num_train_epochs 3 \
  --bf16