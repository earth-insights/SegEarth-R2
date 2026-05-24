import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import re
import cv2
import torch
import numpy as np
from tqdm import tqdm
import transformers
from typing import Optional, List, Tuple
from PIL import Image
from dataclasses import dataclass, field
from transformers import SiglipImageProcessor
import torch.distributed as distributed
import zipfile

from segearth_r2.utils import conversation as conversation_lib
from segearth_r2.utils.builder import load_pretrained_model
from segearth_r2.datasets.dataset import preprocess_image
from segearth_r2.utils.constants import IMAGE_TOKEN_INDEX, REFER_TOKEN_INDEX
from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, LaSeRSDataset, RRSISDDataset, RefSegRSDataset
import torch.nn.functional as F

# ==========================================
# Constants & Configuration
# ==========================================
DEFAULT_PREFIX_INST = (
    "This is an image \n<image>\n. If the instruction is a question, provide segmentation "
    "result + explanation; if it's a declarative phrase, provide only the segmentation result\n. "
    "Please doing Referring or Reasoning Segmentation according to the following instruction:"
)

BASE_COLORS = [
    [255, 0, 0],   
    [0, 255, 0],    
    [0, 0, 255],    
    [255, 255, 0],  
    [255, 0, 255],  
    [0, 255, 255],  
    [255, 165, 0],  
    [128, 0, 128],  
]

@dataclass
class Arguments:
    local_rank: int = 0

    vision_tower: str = "pretrained_model/CLIP"
    vision_tower_mask: str = "pretrained_model/mask2former/model_final_54b88a.pkl"

    lazy_preprocess: bool = False
    base_data_path: Optional[str] = field(default='your_data_path')
    
    model_path: Optional[str] = field(default="SegEarthR2_LaSeRS/hfweights-50000")
    mask_config: Optional[str] = field(default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    model_map_name: str = 'segearth_r2'
    version: str = 'llava_phi'
    
    temperature: float = 0.2
    num_beams: int = 1
    max_new_tokens: int = 128
    do_sample: bool = True

    output_dir: str = 'save_folder'
    dataloader_num_workers: int = 8
# ==========================================
# Helper Functions
# ==========================================
def tokenizer_special_tokens(prompt: str, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, 
                             refer_token_index=REFER_TOKEN_INDEX, return_tensors=None):
    """Tokenize the prompt while preserving special multimodal tokens."""
    input_ids = []
    special_token_map = {'<image>': image_token_index, '<refer>': refer_token_index}
    prompt_chunks = re.split('(<image>|<refer>)', prompt)

    for chunk in prompt_chunks:
        if chunk in special_token_map:
            input_ids.append(special_token_map[chunk])
        elif chunk != '':
            input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
            
    if return_tensors == 'pt':
        return torch.tensor(input_ids, dtype=torch.long).squeeze()
    elif return_tensors is not None:
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
        
    return input_ids

def preprocess_image_clip(image_path: str, clip_image_processor) -> torch.Tensor:
    """Read and preprocess the image specifically for the CLIP vision encoder."""
    img_clip = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_clip = clip_image_processor.preprocess(img_clip, return_tensors="pt")["pixel_values"][0]
    return image_clip

def preprocess_instruction(text: str, prefix_inst: str, tokenizer, conversation_lib) -> torch.Tensor:
    """Format the text instruction into the model's expected conversation template."""
    sources = [[{'from': 'human', 'value': prefix_inst + '\n' + text}, {'from': 'gpt', 'value': ''}]]

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:] # Skip the first one if it is not from human

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"Role mismatch at index {i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = torch.stack(
        [tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0
    )
    return input_ids[0]

def preprocess_input(text: str, image_path: str, tokenizer, clip_image_processor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orchestrate the preprocessing of text and image inputs."""
    instruction = text.strip()

    # Image preprocessing for main model
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    
    image_RGB = preprocess_image(image_path)
    image_tensor = torch.as_tensor(np.ascontiguousarray(image_RGB.transpose(2, 0, 1)))
    images = (image_tensor - pixel_mean) / pixel_std

    # Text and CLIP image preprocessing
    input_ids = preprocess_instruction(instruction, DEFAULT_PREFIX_INST, tokenizer, conversation_lib)
    images_clip = preprocess_image_clip(image_path, clip_image_processor)

    return input_ids.unsqueeze(0), images.unsqueeze(0), images_clip.unsqueeze(0)

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


def main():
    parser = transformers.HfArgumentParser(Arguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    model_path = os.path.expanduser(data_args.model_path)

    print("---------- Initializing Model ----------")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        model_args=data_args, 
        mask_config=data_args.mask_config, 
        device="cuda"
    )
    
    device = torch.device(data_args.local_rank if torch.cuda.is_available() else "cpu") 
    model.to(dtype=torch.float32, device=device)
    # model.eval() # Ensure model is in evaluation mode

    print("---------- Model Initialization Complete ----------")

    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version] # phi-2
    clip_image_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)

    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)
    json_folders = os.path.join(data_args.base_data_path, 'rs_reason_seg/LaSeRS/test/annotations')
    splits = os.listdir(json_folders)
    # save_folder = data_args.output_dir

    for split in splits:
        print(f'------cur benchmark is LaSeRS {split} subset -------')
        eval_dataset = LaSeRSDataset(base_data_path=data_args.base_data_path, tokenizer=tokenizer, data_args=data_args, split=split)
        
        dataloader_params = {
            "batch_size": 1,
            "num_workers": data_args.dataloader_num_workers,
        }

        val_sampler = None
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=dataloader_params['batch_size'],
            shuffle=False,
            num_workers=dataloader_params['num_workers'],
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=data_collator)

        do_eval(model, tokenizer, clip_image_processor, eval_dataloader, split, data_args, device)


def do_eval(model, tokenizer, clip_image_processor, eval_dataloader, split, data_args, device):

    model.eval()
    
    with torch.no_grad():
        SEG_token_id = tokenizer.encode('[SEG]', add_special_tokens=False)[0]
        tokenizer.pad_token = tokenizer.eos_token
        overall_mask_num = 0
        I = 0
        U = 0
        IoU = 0
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            overall_mask_num += inputs['mask_num'][0]
            print(f"masks_num: { inputs['mask_num'][0] }")
            
            text = inputs['seg_info'][0]['instruction']
            image_path = inputs['seg_info'][0]['image_path']
    
            input_ids, images, images_clip = preprocess_input(text, image_path, tokenizer, clip_image_processor)
            output_ids, masks_pred = model.inference(
                input_ids=input_ids.to(device),
                images=images.to(device),
                images_clip=images_clip.to(device),
                do_sample=data_args.do_sample,
                eos_token_id=tokenizer.eos_token_id,
                temperature=data_args.temperature,
                num_beams=data_args.num_beams,
                max_new_tokens=data_args.max_new_tokens,
                use_cache=True, 
                SEG_token_id=SEG_token_id
            )
            
            # input_token_len = input_ids.shape[1]
            # generated_ids = output_ids[0][input_token_len:]
            # output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # text_parts = output_text.split('[SEG]')
            # colored_text = text_parts[0]
            # for i in range(1, len(text_parts)):
            #     r, g, b = BASE_COLORS[(i - 1) % len(BASE_COLORS)]
            #     colored_seg = f"\033[38;2;{r};{g};{b}m[SEG]\033[0m"
            #     colored_text += colored_seg + text_parts[i]

            # print(f"\n Model Input:\n{text}\n")
            # print(f"\n💡 Model Output:\n{colored_text}\n")
            # print("-" * 50)

            gt_masks = []
            for _seg_info in inputs['seg_info']:
                gt_mask = _seg_info['mask'].unsqueeze(0)
                gt_mask = F.interpolate(
                    gt_mask,
                    size=(images.shape[-2], images.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                gt_mask = (gt_mask.cpu().numpy() > 0).astype(np.uint8)
                gt_masks.append(gt_mask) # 每个都是[1, 1, 1024, 1024]的0，1元素

            n_gt = len(gt_masks)
            if masks_pred is None:
                H, W = images.shape[-2], images.shape[-1]
                masks_pred = np.zeros((n_gt, 1, H, W), dtype=np.uint8)
            n_pred = masks_pred.shape[0]

            if n_pred < n_gt:
                masks_pred = np.concatenate(
                    [masks_pred, np.repeat(masks_pred[-1:], n_gt - n_pred, axis=0)], axis=0
                )
            elif n_pred > n_gt:
                masks_pred = masks_pred[:n_gt]
            
            for pred_mask, gt_mask in zip(masks_pred, gt_masks):
                pred_bin = (pred_mask.squeeze(0) > 0).astype(np.uint8)
                gt_bin = gt_mask.squeeze()  # 已是 [H, W] 0/1
                inter = np.logical_and(pred_bin, gt_bin).sum()
                union = np.logical_or(pred_bin, gt_bin).sum()
                I += inter
                U += union
                IoU += (inter / union) if union > 0 else 1.0
                print(f"IoU: {inter / union}")

        print(f"Overall mask num: {overall_mask_num}")
        print(f"Overall pred mask num: {overall_mask_num}")
        print(f"IoU_sum: {IoU}")
        print(f"{split} gIoU: {IoU / overall_mask_num}")

if __name__ == "__main__":
    main()