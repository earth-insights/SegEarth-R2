deepspeed --include localhost:0 --master_port=29500 segearth_r2/eval/eval_inference.py \
    --base_data_path /root/siton-data-412581749c3f4cfea0d7c972b8742057/data \
    --model_path SegEarthR2_LaSeRS/hfweights-50000 \
    --vision_tower_mask pretrained_model/mask2former/model_final_54b88a.pkl \
    --mask_config segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
    --output_dir output/res \