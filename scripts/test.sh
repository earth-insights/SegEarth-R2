deepspeed --include localhost:0 --master_port=29500 segearth_r2/eval/eval.py \
    --base_data_path data_path \
    --model_path model_path \
    --vision_tower_mask pretrained_model/mask2former/model_final_54b88a.pkl \
    --mask_config segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
    --output_dir output/res \
