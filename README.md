# Mug-STAN 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-temporal-modeling-for-clip-based/video-retrieval-on-msr-vtt-1ka)](https://paperswithcode.com/sota/video-retrieval-on-msr-vtt-1ka?p=revisiting-temporal-modeling-for-clip-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-temporal-modeling-for-clip-based/video-retrieval-on-didemo)](https://paperswithcode.com/sota/video-retrieval-on-didemo?p=revisiting-temporal-modeling-for-clip-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-temporal-modeling-for-clip-based/video-retrieval-on-lsmdc)](https://paperswithcode.com/sota/video-retrieval-on-lsmdc?p=revisiting-temporal-modeling-for-clip-based)

Official PyTorch implementation of the paper ["Revisiting Temporal Modeling for CLIP-based Image-to-Video
Knowledge Transferring"](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Revisiting_Temporal_Modeling_for_CLIP-Based_Image-to-Video_Knowledge_Transferring_CVPR_2023_paper.html) and ["Mug-STAN: Adapting Image-Language Pretrained Models for General Video Understanding"](https://arxiv.org/abs/2311.15075)

The original code is based on mmcv1.4. Due to all the data processing pipelines being built on private I/O, the training code cannot be open-sourced. Therefore, we have reproduced the results using mmcv2.0.

## Getting Started
### Installation

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/farewellthree/STAN.git
cd STAN
conda create --name stan python=3.10
conda activate stan
bash install.sh
```

### Prepare Datasets
You can follow [CLIP4clip](https://github.com/ArrowLuo/CLIP4Clip) for the acquisition of videos and annotation.

Once the dataset is already, set the path in each config. Take stan-b/32 on MSRVTT for instance, set video path [here](https://github.com/farewellthree/STAN/blob/main/configs/exp/stan/stan_msrvtt_b32_hf.py#L25) at Line 25.

Considering there might be multiple versions of annotations for the dataset, our code may not be compatible with your annotations. In such cases, you just need to modify the corresponding dataset class in [video_text_dataset.py](https://github.com/farewellthree/STAN/blob/main/mmaction/datasets/video_text_dataset.py), to output the paths of all videos along with their corresponding captions.

## Training
### STAN
To train stan-b/32 on MSRVTT, run 
```bash
torchrun --nproc_per_node=8 --master_port=20001 tools/train.py configs/exp/stan/stan_msrvtt_b32_hf.py --launcher pytorch
```
The same principle applies to other datasets or models in terms of scale.

### Mug-STAN
To train mug-stan-b/32 on MSRVTT, run 
```bash
torchrun --nproc_per_node=8 --master_port=20001 tools/train.py configs/exp/stan/mugstan_msrvt_b32_hf.py --launcher pytorch
```
The same principle applies to other datasets or models in terms of scale.

### Post-Pretraining
To post-pretraining mug-stan-b/32 on Webvid10m, run 
```bash
torchrun --nproc_per_node=16 --master_port=20001 tools/train.py configs/exp/stan/mugstan_webvid10m_b32_pretrain.py --launcher pytorch
```

## Citation
If you find the code useful for your research, please consider citing our paper:
```
@article{liu2023revisiting,
  title={Revisiting Temporal Modeling for CLIP-based Image-to-Video Knowledge Transferring},
  author={Liu, Ruyang and Huang, Jingjia and Li, Ge and Feng, Jiashi and Wu, Xinglong and Li, Thomas H},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
