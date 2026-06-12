# SegEarth-R2: Towards Comprehensive Language-guided Segmentation for Remote Sensing Images

<div align="center">
  <p>
    <a href="https://www.ultralytics.com/events/yolovision?utm_source=github&utm_medium=org&utm_campaign=yv25_event" target="_blank">
      <img width="100%" src="https://github.com/user-attachments/assets/656d9b12-d829-4d7a-8d53-deb19334dc91" alt="Ultralytics YOLO banner"></a>
  </p>
</div>
<br>

- 01/22/2026：比赛的 [训练集](https://pan.baidu.com/s/17EWCQX2bjkXUuAdqISdrLg?pwd=AIRS) 和 [验证集](https://pan.baidu.com/s/146ZLrZcdtpSlWSKW7xH0Dg?pwd=AIRS)已发布。为方便参赛者更快搭建开发环境，现额外提供[Conda环境包](https://pan.baidu.com/s/1Sv9U6alZ5_lPa_6w-EuIAw?pwd=AIRS)。
- 01/21/2026：Baseline代码已发布。
- 01/18/2026：[赛题](https://www.codabench.org/competitions/12624/)已发布，欢迎大家参赛！
- 12/24/2025：LaSeRS数据集将会作为 [**AIRS2026**](www.airs.top)竞赛 中的一部分，因此数据延迟发布。欢迎大家关注！重要信息将会在此界面更新。
- 12/24/2025：[SegEarth-R2](https://arxiv.org/abs/2512.20013) 论文已发布。

## 🔧 Usage：

Follow the guidelines below to set up, train and evaluate:

* [Preparation ⚙️](docs/Preparation.md): Instructions for organizing datasets and pretrained weights for proper model training and inference.
* [Installation 💻](docs/Installation.md): Set up the `segearthr2` conda environment, install dependencies, and clone the repo.
* [Training 🏋️‍♂️](docs/Training.md): Run `scripts/train.sh` with DeepSpeed, modifying parameters like data and model paths for training.
* [Evaluation 🎯](docs/Evaluation.md): Run `scripts/eval.sh` to evaluate the model, updating paths as needed. 

## ⭐️ Citation

If you find this project useful, welcome to cite us.

```bib
@article{xin2025segearth,
  title={SegEarth-R2: Towards Comprehensive Language-guided Segmentation for Remote Sensing Images},
  author={Xin, Zepeng and Li, Kaiyu and Chen, Luodi and Li, Wanchen and Xiao, Yuchen and Qiao, Hui and Zhang, Weizhan and Meng, Deyu and Cao, Xiangyong},
  journal={arXiv preprint arXiv:2512.20013},
  year={2025}
}
```
