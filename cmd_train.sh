cd /afs/zengwang/projects/task_define_service/LLaMA-Factory
export PYTHONPATH=$PYTHONPATH:/afs/zengwang/projects/task_define_service/LLaMA-Factory/src

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
--overwrite_cache \
--overwrite_output_dir \
--logging_steps 10 \
--ddp_timeout 9000  \
--bf16 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--model_name_or_path /afs/zengwang/ckpt/Stream-Qwen2-VL-7B-Instruct \
--template qwen2_vl_stream_v2 \
--freeze_vision_tower \
--finetuning_type lora \
--lora_rank 8 \
--additional_target stream_head,lm_head \
--video_resolution 65536 \
--video_fps 2 \
--video_maxlen 64 \
--cutoff_len 4096 \
--num_train_epochs 1 \
--save_steps 500 \
--stream_loss_factor 2 \
--output_dir work_dirs/stream_v2_lora_2 \
--dataset \
MMDuetIT_dvc_stream_v2_25k,\
MMDuetIT_magqa_stream_v2_25k,\
llava_video_0_30s_academic_mc_stream_v2_5k,\
llava_video_0_30s_academic_oe_stream_v2_10k,\
llava_video_0_30s_academic_cap_stream_v2_10k,\
llava_video_0_30s_youtube_mc_stream_v2_5k,\
llava_video_0_30s_youtube_oe_stream_v2_10k,\
llava_video_0_30s_youtube_cap_stream_v2_10k,\
llava_video_0_30s_stitch_mc_v2_10k,\
llava_video_0_30s_stitch_oe_v2_20k

#    --mask_history \
