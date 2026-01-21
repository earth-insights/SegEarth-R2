# Training🚀

You can run `scripts/train.sh` to start training. Don't forget to modify the parameters in the script to match your own paths.

```bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# ------ main-training ------
# 显存不足时可选择zero3.json或者zero2.json
deepspeed --master_port=29500 --include localhost:4 segearth_r2/train/train.py \
    --model_name_or_path "pretrained_model/mllm/Mipha-3B" \
    --vision_tower "pretrained_model/CLIP/siglip-so400m-patch14-384" \
    --vision_tower_mask "pretrained_model/mask2former/model_final_54b88a.pkl" \
    --base_data_path 'your_data_path' \
    --output_dir output_folder \
    --max_steps 5000 \
    --per_device_train_batch_size 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --bf16 True \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --lora_r 4 \
    --deepspeed scripts/zero3.json \
    --mask_config 'segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml' \
    --data_ratio '1' \
    --switch_bs 4 \
```

After training, you can run `scripts/merge_lora_weights.sh` to merge the LoRA adapter weights into the base model for inference and evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path=your_model_path \
    --vision_tower=pretrained_model/CLIP/siglip-so400m-patch14-384 \
    --vision_tower_mask=pretrained_model/mask2former/model_final_54b88a.pkl \
    --mask_config=segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
    --save_path=your_save_path \
    --lora_r=4
```