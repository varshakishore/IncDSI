# IncDSI

This repo contains official PyTorch implementation of [IncDSI: Incrementally Updatable Document Retrieval](http://proceedings.mlr.press/v202/kishore23a/kishore23a.pdf) (ICML 2023).

## Introduction

## Environment
- Python >= 3.8
- pyTorch >= 1.11
- CUDA >= 11.3
- cuDNN >= 7.6
- tqdm
- datasets >= 1.18.3
- transformers >= 4.20.1
- numpy >= 1.22.4
- wandb >= 0.12.21 (optional)

## Data
The paper presents results on NQ320K and MS MARCO datasets. Download and extract the pre-processed datasets from [here](https://drive.google.com/drive/folders/1JB-DVA3hrk9gIQlTIfRhnGFq5lgZo400?usp=sharing) into `IncDSI/`.

## Instructions

## Citation
```
@article{kishore2023incdsi,
  title={IncDSI: Incrementally Updatable Document Retrieval},
  author={Kishore, Varsha and Wan, Chao and Lovelace, Justin and Artzi, Yoav and Weinberger, Kilian Q},
  year={2023}
}
```
