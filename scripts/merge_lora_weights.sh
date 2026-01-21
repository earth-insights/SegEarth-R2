CUDA_VISIBLE_DEVICES=0 python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path=your_model_path \
    --vision_tower=pretrained_model/CLIP/siglip-so400m-patch14-384 \
    --vision_tower_mask=pretrained_model/mask2former/model_final_54b88a.pkl \
    --mask_config=segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
    --save_path=your_save_path \
    --lora_r=4