# HKNAS: Classification of Hyperspectral Imagery Based on Hyper Kernel Neural Architecture Search (TNNLS 2023)

1dhkcls

CUDA_VISIBLE_DEVICES=0 python main.py \
    --flag 'honghu' --exp_num 10 \
    --block_num 3 --layer_num 1

3dhkcls

CUDA_VISIBLE_DEVICES=0 python main.py \
    --flag 'honghu' --exp_num 1 \
    --block_num 3 --layer_num 3

3dhkseg

CUDA_VISIBLE_DEVICES=0 python main.py \
    --flag 'honghu' --exp_num 3 \
    --block_num 3 --layer_num 1

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
