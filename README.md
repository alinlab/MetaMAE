# Modality-agnostic Self-Supervised Learning with Meta-Learned Masked Auto-Encoder

PyTorch implementation for "[Modality-agnostic Self-Supervised Learning with Meta-Learned Masked Auto-Encoder](https://arxiv.org/abs/2310.16318)" (accepted in NeurIPS 2023)

<img width="100%" src="https://github.com/alinlab/MetaMAE/assets/69646951/ed05afec-a7fd-4ae8-aa68-4ead3f7e8d40" />

**TL;DR:** Interpreting MAE through meta-learning and applying advanced meta-learning techniques to improve unsupervised representation of MAE on arbitrary modalities.


## Install

```bash
conda create -n meta-mae python=3.9
conda activate meta-mae
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
pip install numpy==1.21.5
conda install ignite -c pytorch
pip install timm==0.6.12
pip install librosa
pip install pandas
pip install packaging tensorboard sklearn
```

## Download datasets
- we get the datasets following DABS datasets source codes of official github page: https://github.com/alextamkin/dabs/tree/main/src/datasets
- Our code can be executed only for the preprocessed data with the above source codes (e.g., spliting to make scv, ...)


## Pretraining MetaMAE
- E.g., pamap2
```bash
python pretrain.py --logdir ./logs_final/pamap2/metamae --seed 0 --model metamae \
	--datadir [DATA_ROOT] --dataset pamap2 \
	--inner-lr 0.5 --reg-weight 1 --num-layer-dec 4 --dropout 0.1 --mask-ratio 0.85
```


## Evaluating MetaMAE
```bash
python linear_evaluation.py --ckptdir ./logs_final/pamap2/metamae --seed 0 --model metamae \
	--datadir [DATA_ROOT] --dataset pamap2 \
	--inner-lr 0.5 --reg-weight 1 --num-layer-dec 4 --dropout 0.1 --mask-ratio 0.85
```