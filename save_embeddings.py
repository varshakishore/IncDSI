import torch
from BertModel import QueryClassifier
from train_initial_model import get_dataset, load_dataset_helper, get_dataloader
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
import datasets
import pickle as pkl
import joblib
import argparse
from utils import *


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default='bert-base-uncased',
        choices=['bert-base-uncased'],
        help="Model name",
    )

    parser.add_argument(
        "--dataset", 
        default='NQ320k', 
        choices=['NQ320k','MSMARCO'], 
        help='which dataset to use')

    parser.add_argument(
        "--initialize_model",
        default=None,
        type=str,
        help="path to saved model",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The dir. for log files",
    )

    parser.add_argument(
        "--doc_split",
        default = 'old',
        choices=['old','new', 'tune'],
        help="which split to save"
    )

    parser.add_argument(
        "--split",
        default = 'train',
        choices=['train','val', 'test', 'gen'],
        help="which split to save"
    )

    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="where the train/test/val data is located",
    )
    
    args = parser.parse_args()

    return args


def save(args, model, dataloader, batch_size, dataset_size):

    model.eval()

    embedding = torch.zeros(dataset_size, 768)

    labels = torch.zeros(dataset_size)

    device = torch.device('cuda')

    for i,inputs in enumerate(tqdm(dataloader, desc='forward pass')):
                    
        inputs.to(device)            
        
        with torch.no_grad():
            assert args.model_name == 'bert-base-uncased'
            outputs = model(inputs['input_ids'], inputs['attention_mask'], return_hidden_emb=True)

            if i != len(dataloader) - 1:
                embedding[i*batch_size:(i+1)*batch_size] = outputs.squeeze()
                labels[i*batch_size:(i+1)*batch_size] = inputs['labels']
            else:
                embedding[i*batch_size:i*batch_size+inputs['input_ids'].shape[0],:] = outputs.squeeze()
                labels[i*batch_size:i*batch_size+inputs['input_ids'].shape[0]] = inputs['labels']

    return embedding, labels



def main():
    device = torch.device('cuda')

    args = get_arguments()

    ### HARDCODING 
    # use the same number of class no matter which split to load because the embedding does 
    # not need the classification layer
    class_num = 100000

    assert args.model_name == 'bert-base-uncased'
    model = QueryClassifier(class_num)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='cache')

    if args.dataset == 'NQ320k' or args.dataset == 'MSMARCO':
        data_dirs = {'data': args.base_data_dir,
                    'old': os.path.join(args.base_data_dir, "old_docs"),
                    'tune': os.path.join(args.base_data_dir, "tune_docs"),
                    'new': os.path.join(args.base_data_dir, "new_docs")}
        if args.doc_split in ['old', 'new', 'tune']:
            doc2class = joblib.load(os.path.join(data_dirs[args.doc_split], 'doc_class.pkl'))
        else:
            raise ValueError(f'{args.doc_split} split not supported for {args.dataset} dataset')
        dataset_cls = partial(get_dataset, doc_class=doc2class)
        gen_dataset_cls = partial(get_dataset, doc_class=doc2class)
    else:
        raise ValueError(f'{args.dataset} dataset not supported')

    if args.split == 'gen':
        file_path = os.path.join(data_dirs[args.doc_split], 'passages_seen.json')
        generated_queries = load_dataset_helper(file_path)
        dataset = gen_dataset_cls(tokenizer=tokenizer, datadict = generated_queries)
    else:
        file_path = os.path.join(data_dirs[args.doc_split], f'{args.split}queries.json')
        natural_queries = load_dataset_helper(file_path)
        dataset = dataset_cls(tokenizer=tokenizer, datadict = natural_queries)

    batch_size = 3500
    dataloader = get_dataloader(dataset, batch_size, tokenizer)

    assert args.initialize_model is not None
    load_saved_weights(model, args.initialize_model, load_classifier=False)

    model.to(device)
    embedding_matrix, labels = save(args, model, dataloader, batch_size, len(dataset))
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f'Writing {args.doc_split}-{args.split}-embeddings.pkl')
    joblib.dump(embedding_matrix, os.path.join(args.output_dir,f'{args.doc_split}-{args.split}-embeddings.pkl'))
    print('Done.')
    class2doc = {v:k for k, v in doc2class.items()}
    assert len(class2doc) == len(doc2class)
    doc_ids = torch.tensor([class2doc[i.item()] for i in labels], dtype=torch.long)
    print(f'Writing {args.doc_split}-{args.split}-docids.pkl')
    joblib.dump(doc_ids, os.path.join(args.output_dir, f'{args.doc_split}-{args.split}-docids.pkl'))
    print('Done.')


if __name__ == "__main__":
    main()
