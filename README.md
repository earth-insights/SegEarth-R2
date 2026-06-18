# SegEarth-R2: Towards Comprehensive Language-guided Segmentation for Remote Sensing Images

<div align="center">
  <p>
    <a href="https://www.ultralytics.com/events/yolovision?utm_source=github&utm_medium=org&utm_campaign=yv25_event" target="_blank">
      <img width="100%" src="https://github.com/user-attachments/assets/656d9b12-d829-4d7a-8d53-deb19334dc91" alt="Ultralytics YOLO banner"></a>
  </p>
</div>
<br>

- 05/24/2026: The code is released!
- 05/23/2026: LaSeRS dataset is released on [HuggingFace](https://huggingface.co/datasets/earth-insights/LaSeRS)

## 🔧 Usage：

Follow the guidelines below to set up, train and evaluate:

* [Preparation ⚙️](docs/Preparation.md): Instructions for organizing datasets and pretrained weights for proper model training and inference.
* [Installation 💻](docs/Installation.md): Set up the `segearthr2` conda environment, install dependencies, and clone the repo.
* [Training 🏋️‍♂️](docs/Training.md): Run `scripts/train.sh` with DeepSpeed, modifying parameters like data and model paths for training.
* [Evaluation 🎯](docs/Evaluation.md): Run `scripts/eval.sh` to evaluate the model, updating paths as needed. 

## ⭐️ Citation

If you find this project useful, welcome to cite us.

```bib
@inproceedings{xin2026segearth,
  title={Segearth-r2: Towards comprehensive language-guided segmentation for remote sensing images},
  author={Xin, Zepeng and Li, Kaiyu and Chen, Luodi and Li, Wanchen and Yuchen, Xiao and Qiao, Hui and Zhang, Weizhan and Meng, Deyu and Cao, Xiangyong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13199--13210},
  year={2026}
}
```
