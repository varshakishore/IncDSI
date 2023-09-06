# IncDSI

This repo contains official PyTorch implementation of [IncDSI: Incrementally Updatable Document Retrieval](http://proceedings.mlr.press/v202/kishore23a/kishore23a.pdf) (ICML 2023).

## Introduction
IncDSI, a method to add documents in real time (about 20-50ms per document) to DSI style document retrieval models, without retraining the model on the entire dataset.  The addition of documents is formulated as a constrained optimization
problem that makes minimal changes to the network parameters. Although orders of magnitude faster, this approach is competitive with retraining the model on the whole dataset.


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
The paper presents results on [NQ320K](https://ai.google.com/research/NaturalQuestions) and [MS MARCO](https://microsoft.github.io/msmarco/) datasets. Download and extract the pre-processed datasets from [here](https://drive.google.com/drive/folders/1JB-DVA3hrk9gIQlTIfRhnGFq5lgZo400?usp=sharing) into `IncDSI/data`. For details about how the datasets were pre-processed and split, please refer to the [paper](http://proceedings.mlr.press/v202/kishore23a/kishore23a.pdf).

## Instructions
#### Training an initial DSI style model

#### Caching embeddings for queries 

#### IncDSI

## Citation
```
@article{kishore2023incdsi,
  title={IncDSI: Incrementally Updatable Document Retrieval},
  author={Kishore, Varsha and Wan, Chao and Lovelace, Justin and Artzi, Yoav and Weinberger, Kilian Q},
  year={2023}
}
```
