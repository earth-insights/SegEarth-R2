import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import argparse
import torch
import torch.nn.functional as F
import torch.distributed as distributed
from enum import Enum
import json
from tqdm import tqdm
import numpy as np
import shutil
from tifffile import imwrite as imsave
import zipfile
from PIL import Image

import cv2
from transformers import SiglipImageProcessor

from segearth_r2.utils import conversation as conversation_lib
from segearth_r2.utils.builder import load_pretrained_model
from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, LaSeRSDataset

from dataclasses import dataclass, field
import torch.distributed as dist
import transformers
from typing import Optional


@dataclass
class DataArguments:

    local_rank: int = 0 

    vision_tower: str = "pretrained_model/CLIP/siglip-so400m-patch14-384"
    vision_tower_mask: str = "pretrained_model/mask2former/model_final_54b88a.pkl"

    lazy_preprocess: bool = False
    base_data_path: Optional[str] = field(default='your_data_path')
    model_path: Optional[str] = field(default="your_model_path")
    mask_config: Optional[str] = field(default="../segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    model_map_name: str = 'segearth_r2'
    version: str = 'llava_phi'
    output_dir: str = 'save_folder'
    eval_batch_size: int = 1
    dataloader_num_workers: int = 8

def init_distributed_mode(para):
    para.distributed = True
    if torch.cuda.device_count() <= 1:
        para.distributed = False
        para.local_rank = 0
        para.world_size = 1

    if para.distributed:
         # Init distributed environment
        distributed.init_process_group(backend="nccl")

        local_rank = distributed.get_rank()
        world_size = distributed.get_world_size()
        torch.cuda.set_device(local_rank)
        print('I am rank %d in this world of size %d!' % (local_rank, world_size))
        para.local_rank = local_rank
        para.world_size = world_size

def zip_folder(folder_path):
    folder_path = os.path.abspath(folder_path)
    folder_name = os.path.basename(folder_path)
    zip_path = f"{folder_path}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname=arcname)

def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    init_distributed_mode(data_args)

    model_path = os.path.expanduser(data_args.model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_args=data_args, mask_config=data_args.mask_config, device='cuda')

    device = torch.device(data_args.local_rank if torch.cuda.is_available() else "cpu") 
    model.to(dtype=torch.float32, device=device)

    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    clip_image_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)

    json_folders = os.path.join(data_args.base_data_path, 'test/annotations')
    splits = os.listdir(json_folders)

    save_folder = data_args.output_dir
    os.makedirs(save_folder, exist_ok=True)
    for split in splits:
        if data_args.local_rank == 0:
            print(f'------cur benchmark is LaSeRS {split} subset -------')
        eval_dataset = LaSeRSDataset(base_data_path=data_args.base_data_path, tokenizer=tokenizer, data_args=data_args, split=split)
        dataloader_params = {
            "batch_size": data_args.eval_batch_size,
            "num_workers": data_args.dataloader_num_workers,
        }
        if not data_args.distributed:
            val_sampler = None
        else:
            val_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
        
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=dataloader_params['batch_size'],
            shuffle=False,
            num_workers=dataloader_params['num_workers'],
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=data_collator)
        
        do_eval(model, eval_dataloader, save_folder, split, data_args, device)
    
    zip_folder(save_folder)

def do_eval(model, eval_dataloader, save_folder, split, data_args, device):

    model.eval()

    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            inputs['token_refer_id'] = [ids.to(device) for ids in inputs['token_refer_id']]
            outputs = model.eval_seg(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'].float(),
                images_clip=inputs['images_clip'].float(),
                seg_info=inputs['seg_info'],
                token_refer_id = inputs['token_refer_id'],
                SEG_token_embedding_indices=inputs['SEG_token_embedding_indices'],
                labels=inputs['labels'],
                mask_num=inputs['mask_num']
            )

            for idx, output in enumerate(outputs):
                pred_mask = output['pred']
                image_name = output['image_name']
                id = output['id']
                mask_id = output['mask_id']
                mask_save_name = f"{image_name}_{id}_{split.split('.')[0]}_{mask_id}.tif"
                if pred_mask.ndim > 2:
                    pred_mask = np.squeeze(pred_mask)
                imsave(os.path.join(save_folder, mask_save_name), pred_mask.astype(np.uint8))

if __name__ == "__main__":
    evaluation()