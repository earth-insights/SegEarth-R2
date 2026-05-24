# Evaluation🔍

Run `scripts/eval.sh` to evaluate the model. Make sure to update the script paths to match your setup.

```bash
deepspeed --include localhost:0 --master_port=29500 segearth_r2/eval/eval.py \
    --base_data_path your_data_folder \
    --mask_config segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
    --model_path your_model_path \
    --output_dir save_folder \
    --eval_batch_size 1
```