# [ICLR 2024] Nois Map Guidance: Inversion with Spatial Context for Real Image Editing

<!-- [![arXiv](dd)](dd) -->

> **Nois Map Guidance: Inversion with Spatial Context for Real Image Editing**<br>
> Hansam Cho, Jonghyun Lee, Seoung Bum Kim, Tae-Hyun Oh, Yonghyun Jeong<br>
> 
>**Abstract**: <br>
Text-guided diffusion models have become a popular tool in image synthesis, known for producing high-quality and diverse images. However, their application to editing real images often encounters hurdles primarily due to the text condition deteriorating the reconstruction quality and subsequently affecting editing fidelity. Null-text Inversion (NTI) has made strides in this area, but it fails to capture spatial context and requires computationally intensive per-timestep optimization. Addressing these challenges, we present NOISE MAP GUIDANCE (NMG), an inversion method rich in a spatial context, tailored for real-image editing. Significantly, NMG achieves this without necessitating optimization, yet preserves the editing quality. Our empirical investigations highlight NMG’s adaptability across various editing techniques and its robustness to variants of DDIM inversions.

## Description
Official implementation of Noise Map Guidance: Inversion with Spatial Context for Real Image Editing 

![image](images/teaser.png)

## Setup
```
conda env create -f environment.yaml
conda activate nmg
```

## NMG + Editing Methods

### Prompt-to-Prompt
The [nmg_ptp.ipynb](nmg_ptp.ipynb) is Notebook for NMG with [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626) editing, capable of performing tasks such as  **object swap**, **contextual alterations**, **face attribute editing**, **color change**, and **global editing**. To efficiently process these tasks, it's recommended to use a GPU equipped with a minimum of 15GB of VRAM.

### MasaCtrl
The [nmg_masactrl.ipynb](nmg_masactrl.ipynb) is Notebook for NMG with [MasaCtrl](https://arxiv.org/abs/2304.08465) editing, capable of performing tasks such as  **viewpoint alternation**, and **pose modification**. To efficiently process these tasks, it's recommended to use a GPU equipped with a minimum of 23GB of VRAM.

### pix2pix-zero
The [nmg_pix2pix.ipynb](nmg_pix2pix.ipynb) is Notebook for NMG with [pix2pix-zero](https://arxiv.org/abs/2302.03027) editing, capable of performing tasks such as  **dog &rarr; cat**, and **cat &rarr; dog**. To efficiently process these tasks, it's recommended to use a GPU equipped with a minimum of 11GB of VRAM.

## Acknowledgements
This repository is built upon [diffusers](https://huggingface.co/docs/diffusers/index), unofficial implementation of [prompt-to-prompt](https://github.com/Weifeng-Chen/prompt2prompt/tree/main), [pix2pix-zero pipeline](https://github.com/huggingface/diffusers/blob/v0.16.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py),and [MasaCtrl](https://github.com/TencentARC/MasaCtrl).


<!-- [LDM](https://github.com/CompVis/latent-diffusion), [ControlNet](https://github.com/lllyasviel/ControlNet/tree/main), [Uni-ControlNet](https://github.com/ShihaoZhaoZSH/Uni-ControlNet), and [U<sup>2</sup>-Net](https://github.com/xuebinqin/U-2-Net). ya bois are the real mvps. -->