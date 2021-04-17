# CensNet
Code release for ["Co-embedding of Nodes and Edges with Graph Neural Networks"](https://arxiv.org/abs/2010.13242) (IEEE PAMI 2020)

# Prerequisites
* Python3
* Pytorch == 1.0.0 (with suitable CUDA and CuDNN version)
* Numpy
* argparse
* tqdm

# Dataset
The datasets (Cora, Citeseer and PubMed) are in [GoogleDrive](https://drive.google.com/file/d/1TXVTe2saZ80d26X5zhkqObhfhhTm6vyl/view?usp=sharing) and [BaiduPan (pw:frvg)](https://pan.baidu.com/s/1d5D5qApPvlYVdV5qWlUIgA).  
You need to move the data file in this CenNet file.

# Training
You can run `python train.py` to train and evaluate.

# Citation
If you use this code for you research, please consider citing:  
```
@article{jiang2020co,
  title={Co-embedding of Nodes and Edges with Graph Neural Networks},
  author={Jiang, Xiaodong and Zhu, Ronghang and Li, Sheng and Ji, Pengsheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}
```
