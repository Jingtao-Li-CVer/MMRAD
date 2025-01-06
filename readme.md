## Adapting the segment anything model for multi-modal retinal anomaly detection and localization (Information Fusion2025)

<p align="center">
  <img src=./figs/mmrad.jpg width="600"> 
</p>

This is a PyTorch implementation of the [MMRAD model](https://www.sciencedirect.com/science/article/pii/S1566253524004093): 
```
@article{li2025adapting,
  title={Adapting the segment anything model for multi-modal retinal anomaly detection and localization},
  author={Li, Jingtao and Chen, Ting and Wang, Xinyu and Zhong, Yanfei and Xiao, Xuan},
  journal={Information Fusion},
  volume={113},
  pages={102631},
  year={2025},
  publisher={Elsevier}
}
```

### Outline
1. Segment anything model is applied for the multi-modal retinal disease diagnosis.
2. Anomaly simulation and prompt-tuning strategies are used to fine-tune SAM.
3. A multi-task decoder is proposed for joint anomaly detection and localization.
4. The model can obtain the anomaly score and localize the lesion for each patient.


### Introduction

 &emsp;The fusion of optical coherence tomography (OCT) and fundus modality information can provide a comprehensive diagnosis for retinal artery occlusion (RAO) disease, where OCT provides the cross-sectional examination of the fundus image. Given multi-modal retinal images, an anomaly diagnosis model can discriminate RAO without the need for real diseased samples. Despite this, previous studies have only focused on single-modal diagnosis, because of: 1) the lack of paired modality samples; and 2) the significant imaging differences, which make the fusion difficult with small-scale medical data.

 &emsp;In this paper, we describe how we first built a multi-modal RAO dataset including both OCT and fundus modalities, which supports both the anomaly detection and localization tasks with pixel-level annotation. Motivated by the powerful generalization ability of the recent visual foundation model known as the Segment Anything Model (SAM), we adapted it for our task considering the small-scale property of retinal samples. Specifically, a modality-shared decoder with task-specific tokens is introduced to make SAM support the multi-modal image setting, which includes a mask token for the anomaly localization task at the pixel level and a fusion token for the anomaly detection task at the case level. Since SAM has little medical knowledge and lacks the learning of the “normal” concept, it is infeasible to localize RAO anomalies in the zero-shot manner. To integrate expert retinal knowledge while keeping the general segmentation knowledge, general anomaly simulation for both modalities and a low-level prompt-tuning strategy are introduced.




### Preparation

1. Install required packages according to the requirements.txt.
2. Organize datasets as provided strcture in data folder, where each case corresponds to a folder.
3. Download the [SAM-b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) checkpoint and put in the weights folder.

### Model Training and Testing

1. MMRAD is trained with normal samples only and can infer the anomaly samples directly.
2. Starting the training and testing process using the following one command.

```
python run.py
```


### Qualitative result  

 &emsp;The following are the exemplified localization results of MMRAD on OCT and fundus modalities.

<p align="center">
  <img src=./figs/fp_res.jpg width="600"> 
</p>

<p align="center">
  <img src=./figs/oct_res.jpg width="600"> 
</p>

