# HKNAS: Classification of Hyperspectral Imagery Based on Hyper Kernel Neural Architecture Search (TNNLS 2023)

## Di Wang, Bo Du, Liangpei Zhang and Dacheng Tao

### Pytorch implementation of our paper for Neural Architecture Search based hyperspectral image classification.

<table>
<tr>
<td><img src=Figs/space.png width=565>
<br> 
<figcaption align = "left"><b>Fig.1 - Search Space. </b></figcaption></td>
<td><img src=Figs/algorithm.png width=300>
<br> 
<figcaption align = "right"><b>Fig.2 - Search Algorithm. </b></figcaption> </td>
</tr>
</table>

## Usage
1. Install Pytorch 1.9 with Python 3.8.
2. Clone this repo.
```
git clone https://github.com/DotWang/HKNAS.git
```
3. For **3-D HK-CLS** and **3-D HK-SEG**, setting the 3-D convolution form in ***main.py***
4. Search, Training, Validation, Testing and Predicion (Taka an example of [WHU-Hi-HongHu](http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm) dataset):

- 1-D HK-CLS

```
cd 1DHKCLS
CUDA_VISIBLE_DEVICES=0 python main.py --flag 'honghu' --exp_num 10 --block_num 3 --layer_num 1
```

- 3-D HK-CLS

```
cd 3DHKCLS
CUDA_VISIBLE_DEVICES=0 python main.py --flag 'honghu' --exp_num 10 --block_num 3 --layer_num 3
```

- 3-D HK-SEG

```
cd 3DHKSEG
CUDA_VISIBLE_DEVICES=0 python main.py --flag 'honghu' --exp_num 10 --block_num 3 --layer_num 1
```

## Citation

```
@article{hknas,
  title={HKNAS: Classification of Hyperspectral Imagery Based on Hyper Kernel Neural Architecture Search},
  author={Di Wang, Bo Du, Liangpei Zhang and Dacheng Tao},
  journal={arXiv preprint arXiv:2304.11701},
  year={2023}
}
```

## Relevant Projects
[1] <strong> Pixel and Patch-level Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; Adaptive Spectral–Spatial Multiscale Contextual Feature Extraction for Hyperspectral Image Classification, IEEE TGRS, 2020 | [Paper](https://ieeexplore.ieee.org/document/9121743/) | [Github](https://github.com/DotWang/ASSMN)
<br> <em> &ensp; &ensp;  Di Wang<sup>&#8727;</sup>, Bo Du, Liangpei Zhang and Yonghao Xu</em>

[2] <strong> Image-level/Patch-free Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; Fully Contextual Network for Hyperspectral Scene Parsing, IEEE TGRS, 2021 | [Paper](https://ieeexplore.ieee.org/document/9347487) | [Github](https://github.com/DotWang/FullyContNet)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Bo Du, and Liangpei Zhang</em>
 
[3] <strong> Graph Convolution based Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; Spectral-Spatial Global Graph Reasoning for Hyperspectral Image Classification, IEEE TNNLS, 2023 | [Paper](https://arxiv.org/abs/2106.13952) | [Github](https://github.com/DotWang/SSGRN)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Bo Du, and Liangpei Zhang</em>

[4] <strong> ImageNet Pretraining and Transformer based Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; DCN-T: Dual Context Network with Transformer for Hyperspectral Image Classification, IEEE TIP, 2023 | [Paper](https://arxiv.org/abs/2304.09915) | [Github](https://github.com/DotWang/DCN-T)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Jing Zhang, Bo Du, Liangpei Zhang, and Dacheng Tao</em>
