import copy
import logging

import numpy as np
import json
import torch
import random
import cv2
from PIL import Image

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from pycocotools import mask as coco_mask
from pycocotools.mask import encode, decode, frPyObjects

import torchvision.transforms as transforms

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 0  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence





def _get_dummy_anno(num_classes=-1, has_mask=True):
    anno = {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
    }
    if has_mask:
        anno["segmentation"] = [np.array([0.0] * 6)]
    return anno



def filter_empty_instances_soft(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1 # invalid instances are marked with -1
    return instances

def is_mask_non_empty(rle_mask):
    if rle_mask is None:
        return False
    binary_mask = decode(rle_mask)
    return binary_mask.sum() > 0

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def build_transform_gen(cfg):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    # if cfg.INPUT.RANDOM_FLIP != "none":
    #     augmentation.append(
    #         T.RandomFlip(
    #             horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
    #             vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
    #         )
    #     )

    augmentation.extend([
        # T.ResizeScale(
        #     min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        # ),
        T.ResizeShortestEdge(
            short_edge_length=image_size, max_size=image_size
        ),   
        T.FixedSizeCrop(crop_size=(image_size, image_size), seg_pad_value=0),
    ])

    return augmentation

class IVSDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = build_transform_gen(cfg)
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.sampling_frame_range = 10
        self.sampling_interval = 1
        self.sampling_frame_num = 3
        self.sampling_frame_shuffle = False

        # 定义resize transform
        self.resize_transform = transforms.Resize((640, 640), interpolation=Image.NEAREST)


    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def preprocess(self, dataset_dict, region_mask_type=None,clip_image_processor=None, mask_format='polygon', crop_frame=False, data_aug=False):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if isinstance(dataset_dict["file_name"],str):
            image = utils.read_image(dataset_dict["file_name"], format='RGB')
        else:
            image = np.array(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)


        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["image"] = (image - self.pixel_mean) / self.pixel_std
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        dataset_dict['transforms'] = clip_image_processor


        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            annotations = dataset_dict['annotations']

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            if len(annos) ==0:
                print('error')


            filter_annos = annos
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            # instances = utils.annotations_to_instances(annos, image_shape)
            instances = utils.annotations_to_instances(filter_annos, image_shape, mask_format='bitmask')
            if 'category_id' in filter_annos[0]:
                classes = [obj["category_id"] for obj in filter_annos]
                instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            # non_empty_instance_mask = [len(obj.get('segmentation', [])) > 0 for obj in annos]
            non_empty_instance_mask = [len(obj.get('segmentation', [])) > 0 for obj in filter_annos]

            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                if hasattr(gt_masks,'polygons'):
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                else:
                    gt_masks = gt_masks.tensor.to(dtype=torch.uint8)
                instances.gt_masks = gt_masks
            # instances.gt_boxes = get_boxes(instances, needing_convert=True)

            dataset_dict["instances"] = instances

        return dataset_dict



    def preprocess_reason(self, dataset_dict, mask_format='polygon', data_aug=False, is_train=True):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = utils.read_image(dataset_dict["file_name"], format='RGB')

        json_path = dataset_dict['json_path'] # .replace(".json", '_new.json')
        masks, sents, is_sentence = get_mask_from_json(json_path, image)

        dataset_dict['sentences'] = sents
        dataset_dict['height'], dataset_dict['width'] = image.shape[:2]
        


        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])


        image, transforms = T.apply_transform_gens(self.tfm_gens, image)


        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["image"] = (image - self.pixel_mean) / self.pixel_std
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        instances = Instances(image_shape)
        if is_train:
            masks = transforms.apply_segmentation(masks)
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in [masks]])
        )
        instances.gt_masks = masks
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        instances.gt_masks = masks.tensor.to(dtype=torch.uint8)

        classes = torch.tensor([1], dtype=torch.int64)
        instances.gt_classes = classes
            
        dataset_dict["instances"] = instances



        return dataset_dict




    def preprocess_refvos(self, ori_dataset_dict, is_train=True, clip_image_processor=None, sample_frame_num=1, reference_frame_num=0):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        
        video_length = ori_dataset_dict["length"]

        trials = 0
        max_trials_nums = 200

        sample_frame_num_all = sample_frame_num + reference_frame_num
        
        while trials < max_trials_nums:
            flag = False

            dataset_dict = copy.deepcopy(ori_dataset_dict)  # it will be modified by code below

            if is_train:

                if sample_frame_num_all > 1:
                    try:
                        sampling_frame_range = np.random.randint(2, 10)
                        this_max_jump = min(video_length, sampling_frame_range)
                        frames_idx = [np.random.randint(video_length)]
                        acceptable_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(video_length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
                        while(len(frames_idx) < sample_frame_num_all):
                            idx = np.random.choice(list(acceptable_set))
                            frames_idx.append(idx)
                            new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(video_length, frames_idx[-1]+this_max_jump+1)))
                            acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))
                        selected_idx = sorted(frames_idx)
                        if np.random.rand() < 0.3:
                            # Reverse time
                            selected_idx = selected_idx[::-1]
                    except:
                        return None

                else:
                    ref_frame = random.randrange(video_length)
                    selected_idx = [ref_frame]
            else:
                selected_idx = range(video_length)

            # selected_idx is a List of length self.sampling_frame_num

            if is_train: 
                video_annos = dataset_dict.pop("annotations", None) # List
                dataset_dict["instances"] = []
            file_names = dataset_dict.pop("file_names", None) # List

            if is_train:
                _ids = set()
                for frame_idx in selected_idx:
                    _ids.update([anno["id"] for anno in video_annos[frame_idx]])
                ids = dict()
                for i, _id in enumerate(_ids):
                    ids[_id] = i # original instance id -> zero-based
            
            dataset_dict["image"] = []
            dataset_dict["padding_mask"] = []
            dataset_dict["file_name"] = []
            dataset_dict['transforms'] = []

            half_reference_frame_num = reference_frame_num // 2

            if not is_train:
                sample_frame_num = video_length
                half_reference_frame_num = 0

            
            for frame_idx in selected_idx[half_reference_frame_num:half_reference_frame_num+sample_frame_num]:
                dataset_dict["file_name"].append(file_names[frame_idx])

                # Read image
                image = utils.read_image(file_names[frame_idx], format='RGB')
                try:
                    utils.check_image_size(dataset_dict, image)
                except:
                    # there are some videos with inconsistent resolutions...
                    # eg. GOT10K/val/GOT-10k_Val_000137
                    print(f'wrong image file: {file_names[frame_idx]}')
                    return None
                
                origin_image_shape = image.shape[:2]
                padding_mask = np.ones(image.shape[:2])
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                # the crop transformation has default padding value 0 for segmentation
                padding_mask = transforms.apply_segmentation(padding_mask)
                padding_mask = ~ padding_mask.astype(bool)


                image_shape = image.shape[:2]  # h, w
                # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
                # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
                # Therefore it's important to use torch.Tensor.
                image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
                dataset_dict["image"].append((image - self.pixel_mean) / self.pixel_std)
                dataset_dict["padding_mask"].append(torch.as_tensor(np.ascontiguousarray(padding_mask)))
                dataset_dict['transforms'].append(transforms)

                # for SOT and VOS, we need the box anno in the 1st frame during inference
                # if (video_annos is None) or (not is_train):
                #     continue
                if not is_train:

                    continue

                # NOTE copy() is to prevent annotations getting changed from applying augmentations
                _frame_annos = []
                for anno in video_annos[frame_idx]:
                    _anno = {}
                    for k, v in anno.items():
                        _anno[k] = copy.deepcopy(v)
                    _frame_annos.append(_anno)

                has_mask = dataset_dict["has_mask"]
                # USER: Implement additional transformations if you have other types of data
                
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in _frame_annos
                ]
                sorted_annos = [_get_dummy_anno(has_mask=has_mask) for _ in range(len(ids))]

                for _anno in annos:
                    idx = ids[_anno["id"]]
                    sorted_annos[idx] = _anno
                _gt_ids = [_anno["id"] for _anno in sorted_annos]

                
                instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
        

                instances.gt_ids = torch.tensor(_gt_ids)
                instances_tmp = utils.filter_empty_instances(copy.deepcopy(instances))
                if len(instances_tmp) == 0:
                    trials += 1
                    # print(f'------trials 1 : {trials}-----')
                    flag = True
                else:
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                    instances = filter_empty_instances_soft(instances)
                    if instances.has("gt_masks"):
                        instances.gt_masks = instances.gt_masks.tensor.to(dtype=torch.uint8)


                    if torch.sum(instances.gt_ids != -1) == 0:
                        trials += 1
                        # print(f'------trials 2 : {trials}-----')
                        flag = True
                    dataset_dict["instances"].append(instances)

            if not flag:

                dataset_dict["image"] = torch.stack(dataset_dict["image"], dim=0)
                dataset_dict["padding_mask"] = torch.stack(dataset_dict["padding_mask"], dim=0)

                
                # append refernece frames
                if is_train:
                    left_selected_idx = selected_idx[:half_reference_frame_num] + selected_idx[half_reference_frame_num+sample_frame_num:]
                    for frame_idx in left_selected_idx:
                        dataset_dict["file_name"].append(file_names[frame_idx])

                if not is_train or sample_frame_num == 1:
                    return dataset_dict
                
                cur_flag = False
                for k in range(sample_frame_num):
                    final_ref_instance = dataset_dict["instances"][k]
                    ref_masks = final_ref_instance.gt_masks
                    for m in ref_masks:
                        if m.nonzero().shape[0] <= 0:
                            trials += 1
                            cur_flag = True
                            # print(f'------trials 4 : {trials}-----')
                            break
                if not cur_flag:
                    return dataset_dict


    def preprocess_revos(self, ori_dataset_dict, ori_select_object, is_train=True, data_aug=False, clip_image_processor=None, sample_frame_num=1, reference_frame_num=0):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        
        video_length = ori_dataset_dict["length"]

        # assert video_length == len(ori_select_object['anno'][0])

        trials = 0
        max_trials_nums = 200

        sample_frame_num_all = sample_frame_num + reference_frame_num
        
        while trials < max_trials_nums:
            flag = False

            dataset_dict = copy.deepcopy(ori_dataset_dict)  # it will be modified by code below
            select_object = copy.deepcopy(ori_select_object)
                
            if sample_frame_num_all > 1:
                try:
                    sampling_frame_range = np.random.randint(2, 10)
                    this_max_jump = min(video_length, sampling_frame_range)
                    frames_idx = [np.random.randint(video_length)]
                    acceptable_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(video_length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
                    while(len(frames_idx) < sample_frame_num_all):
                        idx = np.random.choice(list(acceptable_set))
                        frames_idx.append(idx)
                        new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(video_length, frames_idx[-1]+this_max_jump+1)))
                        acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))
                    selected_idx = sorted(frames_idx)
                    if np.random.rand() < 0.3:
                        # Reverse time
                        selected_idx = selected_idx[::-1]
                except:
                    return None

            else:
                ref_frame = random.randrange(video_length)
                selected_idx = [ref_frame]

            # selected_idx is a List of length self.sampling_frame_num


            video_annos = select_object.pop("anno", None) # List

            dataset_dict["instances"] = []
            file_names = dataset_dict.pop("file_names", None) # List

            
            dataset_dict["image"] = []
            dataset_dict["padding_mask"] = []
            dataset_dict["file_name"] = []
            dataset_dict['transforms'] = []

            half_reference_frame_num = reference_frame_num // 2

            for frame_idx in selected_idx[half_reference_frame_num:half_reference_frame_num+sample_frame_num]:
                dataset_dict["file_name"].append(file_names[frame_idx])

                # Read image
                image = utils.read_image(file_names[frame_idx], format='RGB')
                try:
                    utils.check_image_size(dataset_dict, image)
                except:
                    # there are some videos with inconsistent resolutions...
                    # eg. GOT10K/val/GOT-10k_Val_000137
                    print(f'wrong image file: {file_names[frame_idx]}')
                    return None
                
                origin_image_shape = image.shape[:2]
                padding_mask = np.ones(image.shape[:2])
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                # the crop transformation has default padding value 0 for segmentation
                padding_mask = transforms.apply_segmentation(padding_mask)
                padding_mask = ~ padding_mask.astype(bool)


                image_shape = image.shape[:2]  # h, w
                # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
                # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
                # Therefore it's important to use torch.Tensor.
                image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
                dataset_dict["image"].append((image - self.pixel_mean) / self.pixel_std)
                dataset_dict["padding_mask"].append(torch.as_tensor(np.ascontiguousarray(padding_mask)))
                dataset_dict['transforms'].append(transforms)

                m_final = np.zeros(origin_image_shape, dtype=np.uint8)
                for anno in video_annos:
                    segm = anno[frame_idx]
                    if segm is not None:
                        m = decode(segm)
                        if m.ndim == 3:
                            m = m.sum(axis=2).astype(np.uint8)
                        else:
                            m = m.astype(np.uint8)
                        m_final = m_final | m

                # USER: Implement additional transformations if you have other types of data
                
                instances = Instances(image_shape)
                if is_train:
                    masks = transforms.apply_segmentation(m_final)
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in [masks]])
                )
                for m in masks.tensor:
                    if m.nonzero().shape[0] <= 0:
                        trials += 1
                        # print(f'------trials 1 : {trials}-----')
                        flag = True
                
                if flag:
                    continue

                else:
                    instances.gt_masks = masks
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                    instances.gt_masks = masks.tensor.to(dtype=torch.uint8)

                    classes = torch.tensor([1], dtype=torch.int64)
                    instances.gt_classes = classes

                    dataset_dict["instances"].append(instances)

            if not flag:

                dataset_dict["image"] = torch.stack(dataset_dict["image"], dim=0)
                dataset_dict["padding_mask"] = torch.stack(dataset_dict["padding_mask"], dim=0)

                left_selected_idx = selected_idx[:half_reference_frame_num] + selected_idx[half_reference_frame_num+sample_frame_num:]
                # append refernece frames
                for frame_idx in left_selected_idx:
                    dataset_dict["file_name"].append(file_names[frame_idx])

                return dataset_dict

                # if not is_train or sample_frame_num == 1:
                #     return dataset_dict
                
                # cur_flag = False
                # for k in range(sample_frame_num):
                #     final_ref_instance = dataset_dict["instances"][k]
                #     ref_masks = final_ref_instance.gt_masks
                #     for m in ref_masks:
                #         if m.nonzero().shape[0] <= 0:
                #             trials += 1
                #             cur_flag = True
                #             print(f'------trials 2 : {trials}-----')
                #             break
                # if not cur_flag:
                #     return dataset_dict


    def preprocess_revos_test(self, ori_dataset_dict, is_train=False, clip_image_processor=None,):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        
        video_length = ori_dataset_dict["length"]


        dataset_dict = copy.deepcopy(ori_dataset_dict)  # it will be modified by code below
        
        selected_idx = range(video_length)

        # selected_idx is a List of length self.sampling_frame_num

        dataset_dict["instances"] = []
        file_names = dataset_dict.pop("file_names", None) # List

        
        dataset_dict["image"] = []
        dataset_dict["padding_mask"] = []
        dataset_dict["file_name"] = []
        dataset_dict['transforms'] = []

        for frame_idx in selected_idx:
            dataset_dict["file_name"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format='RGB')
            try:
                utils.check_image_size(dataset_dict, image)
            except:
                # there are some videos with inconsistent resolutions...
                # eg. GOT10K/val/GOT-10k_Val_000137
                print(f'wrong image file: {file_names[frame_idx]}')
                return None
            
            origin_image_shape = image.shape[:2]
            padding_mask = np.ones(image.shape[:2])
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            # the crop transformation has default padding value 0 for segmentation
            padding_mask = transforms.apply_segmentation(padding_mask)
            padding_mask = ~ padding_mask.astype(bool)


            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict["image"].append((image - self.pixel_mean) / self.pixel_std)
            dataset_dict["padding_mask"].append(torch.as_tensor(np.ascontiguousarray(padding_mask)))
            dataset_dict['transforms'].append(transforms)

        dataset_dict["image"] = torch.stack(dataset_dict["image"], dim=0)
        dataset_dict["padding_mask"] = torch.stack(dataset_dict["padding_mask"], dim=0)

        return dataset_dict

def build_transform_gen_for_eval(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    # if cfg.INPUT.RANDOM_FLIP != "none":
    #     augmentation.append(
    #         T.RandomFlip(
    #             horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
    #             vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
    #         )
    #     )

    augmentation.extend([
        T.ResizeShortestEdge(
            short_edge_length=image_size, max_size=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size), seg_pad_value=0),
    ])

    return augmentation


