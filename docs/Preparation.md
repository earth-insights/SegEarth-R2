# Dataset🚀

Download competition dataset from [here](https://www.codabench.org/competitions/12624/)

```
data_path/
├── train/
│   └── annotations/
│   |    ├── train_ann.json
│   └──images/
| 		 └── ...
|
├── test/
|   └──annotations/
|        ├── ... (仅包含问题（description）和答案（answer）)
|  	└──images/
|        └── ...
```

# Pretrained Weights📂

You can download the pre-trained weights of [Mipha-3B](https://huggingface.co/zhumj34/Mipha-3B), [CLIP](https://huggingface.co/google/siglip-so400m-patch14-384) and [Mask2Former](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_54b88a.pkl) from these links, and place them in the `pretrained_model` folder according to the following structure:

```
pretrained_model/
├── CLIP/siglip-so400m-patch14-384/
│    └── ...
├── mask2former/
│    └── model.pkl
├── mllm/Mipha-3B/
     └── ... 
```
