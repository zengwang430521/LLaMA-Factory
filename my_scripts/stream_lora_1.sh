cd /afs/zengwang/projects/task_define_service/LLaMA-Factory

NPROC_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "
echo "DISTRIBUTED_ARGS:$DISTRIBUTED_ARGS"

torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --learning_rate 1.0e-5 \
    --lr_scheduler_type cosine \
    --stage sft \
    --do_train \
    --model_name_or_path /afs/zengwang/ckpt/Stream-Qwen2-VL-7B-Instruct \
    --template qwen2_vl_stream \
    --overwrite_cache \
    --overwrite_output_dir \
    --logging_steps 10 \
    --ddp_timeout 9000  \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --mask_history \
    --freeze_vision_tower \
    --finetuning_type lora \
    --lora_rank 8 \
    --additional_target stream_head \
    --bf16 \
    --video_resolution 65536 \
    --video_fps 2 \
    --video_maxlen 64 \
    --cutoff_len 4096 \
    --dataset MMDuetIT_dvc_stream,MMDuetIT_magqa_stream \
    --image_dir /afs/zengwang/projects/task_define_service/data/shot2story-videos_release_134k \
    --num_train_epochs 1 \
    --save_steps 500 \
    --output_dir work_dirs/stream_lora_1
