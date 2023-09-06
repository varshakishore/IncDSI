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
#### Training an initial DSI model
Train a DSI document retrieval model on the set of old documents. 
```bash
cd IncDSI
bash train_initial_model.sh "$(pwd)/data/dataset_name" "$(pwd)/output/initial_model/"
```
The dataset_name can be "NQ320k" or "MSMARCO." The first path points to the data directory and the second path points to the ouput directory. 

#### Caching embeddings for queries 
The query embeddings for the old and new documents are cached by using the embeddings from the model trained in the step above.
```bash
cd IncDSI
bash save_embeddings.sh "$(pwd)/output/initial_model/base_model_epoch20" "$(pwd)/output/saved_embeddings/" "NQ320K"
```
Note: If the initial models is not trained for 20 epochs, update the first path to point to the correct final model.

#### IncDSI
IncDSI is used to add new documents. 
```bash
cd IncDSI
bash incdsi.sh "$(pwd)/data/dataset_name" "$(pwd)/output/saved_embeddings/" "$(pwd)/output/initial_model/base_model_epoch20" "$(pwd)/output/final_incdsi_model/"
```

## Citation
```
@article{kishore2023incdsi,
  title={IncDSI: Incrementally Updatable Document Retrieval},
  author={Kishore, Varsha and Wan, Chao and Lovelace, Justin and Artzi, Yoav and Weinberger, Kilian Q},
  year={2023}
}
```
