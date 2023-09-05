import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from transformers import T5Tokenizer, BertTokenizer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from dataclasses import dataclass
from BertModel import QueryClassifier
import random
import logging
logger = logging.getLogger(__name__)
import argparse
import os
import joblib
from utils import *

import wandb

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        default=50,
        type=int,
        required=False,
        help="batch_size",
    )

    parser.add_argument(
        "--train_epochs",
        default=128,
        type=int,
        help="Number of train epochs",
    )

    parser.add_argument(
        "--model_name",
        default='bert-base-uncased',
        choices=['bert-base-uncased'],
        help="Model name",
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        required=False,
        help="random seed",
    )

    parser.add_argument(
        "--learning_rate",
        default=5e-3,
        type=float,
        help="initial learning rate for Adam",
    )

    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="only runs validaion",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The dir. for log files",
    )

    parser.add_argument(
        "--logging_step",
        default=50,
        type=int,
        required=False,
        help="steps to log train loss and accuracy"
    )

    parser.add_argument(
        "--freeze_base_model",
        action="store_true",
        help="for freezing the parameters of the base model",
    )

    parser.add_argument(
        "--initialize_model",
        default=None,
        type=str,
        help="path to saved model",
    )

    parser.add_argument(
        "--base_data_dir_new",
        default=None,
        help="finetune with old and new documents",
    )

    parser.add_argument(
        "--base_data_dir",
        type=str,
        default="/home/vk352/dsi/data/NQ320k",
        help="where the train/test/val data is located",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="base_model_epoch",
        help="name for savecd model",
    )

    parser.add_argument(
        "--test_only",
        action="store_true",
        help="run eval on test set",
    )

    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="name for wandb",
    )

    parser.add_argument(
        "--filter_num",
        type=int,
        default=-1,
        help="num new docs",
    )
    

    args = parser.parse_args()

    return args

@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)
                
        inputs['labels'] = torch.Tensor(docids).long()
        return inputs
    
class get_dataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            datadict,
            doc_class):
        super().__init__()
        self.train_data = datadict
        
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.doc_class = doc_class


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.train_data[idx]

        text = [data[key] for key in ['question', 'doc_text', 'gen_question'] if key in data.keys()]
        assert len(text) == 1, "More than one text field in data"

        input_ids = self.tokenizer(text[0],
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=32).input_ids[0]
        
        return input_ids, self.doc_class[data['doc_id']]
    
def get_dataloader(dataset, batch_size, tokenizer, padding='longest', shuffle=False, drop_last=False):
    return DataLoader(dataset, 
        batch_size=batch_size,
        collate_fn=IndexingCollator(
        tokenizer,
        padding=padding),
        shuffle=shuffle,
        drop_last=drop_last)

    
def train(args, model, train_dataloader, optimizer, length):
    """" Training loop for the model. """

    model.train()
        
    total_correct_predictions = 0
    tr_loss = 0

    device = torch.device('cuda')
    loss_func  = torch.nn.CrossEntropyLoss()

    for i, inputs in enumerate(train_dataloader):

        inputs.to(device)

        decoder_input_ids = torch.zeros((inputs['input_ids'].shape[0],1))

        logits = model(inputs['input_ids'], inputs['attention_mask'])
        _, docids = torch.max(logits, 1)
        
        loss = loss_func(logits,torch.tensor(inputs['labels']).long())

        correct_cnt = (docids == inputs['labels']).sum()
        
        tr_loss += loss.item()
        
        total_correct_predictions += correct_cnt

        if (i + 1) % args.logging_step == 0:
            logger.info(f'Train step: {i}, loss: {(tr_loss/i)}')
            if args.wandb_name:
                wandb.log({'train_loss': tr_loss/i})

        loss.backward()
        optimizer.step()
        model.zero_grad()
        # global_step += 1

    
    correct_ratio = float(total_correct_predictions / length) 
    

    logger.info(f'Train accuracy:{correct_ratio}')
    logger.info(f'Loss:{tr_loss/(i+1)}')
    return correct_ratio, tr_loss

def validate(args, model, val_dataloader):
    """" Validating loop for the model. """

    model.eval()

    hit_at_1 = 0
    hit_at_10 = 0
    mrr_at_10 = 0
    hit_at_5 = 0
    val_loss = 0

    device = torch.device('cuda')


    for i,inputs in tqdm(enumerate(val_dataloader), desc='Evaluating dev queries'):
                    
        inputs.to(device)
        loss_func  = torch.nn.CrossEntropyLoss()            
        
        with torch.no_grad():
                            
            decoder_input_ids = torch.zeros((inputs['input_ids'].shape[0],1))
            decoder_input_ids = decoder_input_ids.to(device)
            
            logits = model(inputs['input_ids'], inputs['attention_mask'])
                        
            loss = loss_func(logits,torch.tensor(inputs['labels']).long())

            val_loss += loss.item()
            
            _, docids = torch.max(logits, 1)

            hit_at_1 += (docids == inputs['labels']).sum()


            max_idxs_5 = torch.argsort(logits, 1, descending=True)[:, :5]
            hit_at_5 += (max_idxs_5 == inputs['labels'].unsqueeze(1)).any(1).sum()

            # compute recall@10
            max_idxs_10 = torch.argsort(logits, 1, descending=True)[:, :10]
            hit_at_10 += (max_idxs_10 == inputs['labels'].unsqueeze(1)).any(1).sum()

            #compute mrr@10. Sum will later be divided by number of elements

            mrr_at_10 += (1/ (torch.where(max_idxs_10 == inputs['labels'][:, None])[1] + 1)).sum()

    
    logger.info(f'Validation Loss: {val_loss/(i+1)}')
            
            
    return hit_at_1, hit_at_5, hit_at_10, mrr_at_10

def validate_script(args, tokenizer, model, doc_type=None, split=None, filter_doc_list=None, permutation_seed=None):

    device = torch.device("cuda")

    logging.info(f'Device: {device}')

    if doc_type == "old":
        data_dir = os.path.join(args.base_data_dir, 'old_docs')
        doc_class = joblib.load(os.path.join(data_dir, 'doc_class.pkl'))
    elif doc_type == "new":
        data_dir = os.path.join(args.base_data_dir, 'new_docs')
        if permutation_seed is None:
            doc_class = joblib.load(os.path.join(data_dir, 'doc_class.pkl'))
        else:
            doc_class = joblib.load(os.path.join('/home/jl3353/dsi/data/NQ320k/new_docs', f'doc_class_seed{permutation_seed}.pkl'))
        if 'MSMARCO' in args.base_data_dir:
            doc_list = joblib.load(os.path.join(data_dir, 'doc_list.pkl'))
            doc_list = doc_list[:10000]
    elif doc_type == "tune":
        data_dir = os.path.join(args.base_data_dir, 'tune_docs')
        doc_class = joblib.load(os.path.join(data_dir, 'doc_class.pkl'))
        if 'MSMARCO' in args.base_data_dir:
            doc_list = joblib.load(os.path.join(data_dir, 'doc_list.pkl'))
            doc_list = doc_list[:1000]
    else:
        raise ValueError(f'doc_type={doc_type} must be old, new, or tune')

    if split == "train":
        data = load_dataset_helper(os.path.join(data_dir, 'trainqueries.json'))
        print('train set loaded')

    elif split == "valid":
        data = load_dataset_helper(os.path.join(data_dir, 'valqueries.json'))
        print('validation set loaded')

    elif split == "test":
        data = load_dataset_helper(os.path.join(data_dir, 'testqueries.json'))
        print('test set loaded')

    elif split == "seenq":
        data = load_dataset_helper(os.path.join(data_dir, 'passages_seen.json'))
        print('seen generated queries loaded')

    elif split == "unseenq":
        data = load_dataset_helper(os.path.join(data_dir, 'passages_unseen.json'))
        print('unseen generated queries loaded')
    else:
        raise ValueError(f'split={split} must be train, valid, test, seenq, or unseenq')
    
    if 'MSMARCO' in args.base_data_dir and doc_type in {'new', 'tune'}:
        data = data.filter(lambda example: example['doc_id'] in doc_list)
    if filter_doc_list is not None:
        data = data.filter(lambda example: example['doc_id'] in filter_doc_list)
        if len(data) == 0:
            return None, None, None, None

    if split == "train" or split == "valid" or split == 'test':
        dataset =  get_dataset(tokenizer=tokenizer, datadict = data, doc_class = doc_class)

    elif split == "seenq" or split == "unseenq":
        dataset = get_dataset(tokenizer=tokenizer, datadict = data, doc_class = doc_class)

    dataloader = get_dataloader(dataset, args.batch_size, tokenizer)

    hits_at_1, hits_at_5, hits_at_10, mrr_at_10 = validate(args, model, dataloader)
    length = len(dataloader.dataset)
    hits_at_1 = hits_at_1/length
    hits_at_5 = hits_at_5/length
    hits_at_10 = hits_at_10/length
    mrr_at_10 = mrr_at_10/length

    return hits_at_1, hits_at_5, hits_at_10, mrr_at_10

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataset_helper(path):
    data = datasets.load_dataset(
    'json',
    data_files=path,
    ignore_verifications=False,
    cache_dir='cache'
    )['train']

    return data

def log_metrics(hit_at_1, hit_at_5, hit_at_10, mrr_at_10, length, wandb_log=None, i=0, ext=""):
    logger.info(f'Accuracy: {hit_at_1} / {length} = {hit_at_1/length}')
    logger.info(f'Hits@5: {hit_at_5} / {length} = {hit_at_5/length}')
    logger.info(f'Hits@10: {hit_at_10} / {length} = {hit_at_10/length}')
    logger.info(f'MRR@10: {mrr_at_10} / {length} = {mrr_at_10/length}')

    if wandb_log:
        wandb.log({f'Hits@1_{ext}': hit_at_1/length, f'Hits@5_{ext}': hit_at_5/length, \
            f'Hits@10_{ext}': hit_at_10/length, f'MRR@10_{ext}': mrr_at_10/length, "epoch": i+1})

def main():

    args = get_arguments()

    set_seed(args.seed)

    if not args.validate_only or not args.test_only:
        if args.wandb_name:
            wandb.init(project="IncDSI", name=args.wandb_name)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    )

    logging.basicConfig(filename=f'{args.output_dir}/out.log', encoding='utf-8', level=logging.DEBUG)

    device = torch.device("cuda")

    logging.info(f'Device: {device}')

    train_data = load_dataset_helper(os.path.join(args.base_data_dir, 'old_docs', 'trainqueries.json'))
    generated_queries = load_dataset_helper(os.path.join(args.base_data_dir, 'old_docs', 'passages_seen.json'))
    val_data = load_dataset_helper(os.path.join(args.base_data_dir, 'old_docs', 'valqueries.json'))

    logger.info('train, generated, val loaded')

    if args.base_data_dir_new or args.test_only:
        train_data_new = load_dataset_helper(os.path.join(args.base_data_dir_new, 'trainqueries.json'))
        generated_queries_new = load_dataset_helper(os.path.join(args.base_data_dir_new, 'passages_seen.json'))
        val_data_new = load_dataset_helper(os.path.join(args.base_data_dir_new, 'valqueries.json'))


        logger.info('new train, generated, val set loaded')

    if args.test_only:
        test_data = load_dataset_helper(os.path.join(args.base_data_dir, 'old_docs', 'testqueries.json'))
        test_data_new = load_dataset_helper(os.path.join(args.base_data_dir_new, 'testqueries.json'))

        logger.info('test set loaded')

    old_docs_list = joblib.load(os.path.join(args.base_data_dir, 'old_docs', 'doc_list.pkl'))
    class_num = len(old_docs_list)
    if args.base_data_dir_new or args.test_only:
        new_docs_list = joblib.load(os.path.join(args.base_data_dir_new, 'doc_list.pkl'))

        if args.filter_num!=-1:
            filter_docs = new_docs_list[:args.filter_num]
            train_data_new = train_data_new.filter(lambda example: example['doc_id'] in filter_docs)
            generated_queries_new = generated_queries_new.filter(lambda example: example['doc_id'] in filter_docs)
            val_data_new = val_data_new.filter(lambda example: example['doc_id'] in filter_docs)
            if args.test_only:
                test_data_new = test_data_new.filter(lambda example: example['doc_id'] in filter_docs)
            class_num += args.filter_num
        else:
            class_num += len(new_docs_list)

    logger.info(f'Class number {class_num}')

    logger.info(f'Loading Model and Tokenizer for {args.model_name}')

    model = QueryClassifier(class_num)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='cache')

    # load saved model if initialize_model
    # TODO extend for T5
    if args.initialize_model:
        load_saved_weights(model, args.initialize_model, strict_set=False)

    # TODO check what the name in the T5 model is
    if args.freeze_base_model:
        for name, param in model.named_parameters():
            if name !='classifier.weight':
                param.requires_grad = False
            else:
                print("Unfrozen weight:", name)

    model.to(device)

    logger.info('model loaded')

    doc_class = joblib.load(os.path.join(args.base_data_dir, 'old_docs', 'doc_class.pkl'))

    natural_queries = get_dataset(tokenizer=tokenizer, datadict = train_data, doc_class=doc_class)
    val_queries = get_dataset(tokenizer=tokenizer, datadict = val_data, doc_class=doc_class)
    gen_queries = get_dataset(tokenizer=tokenizer, datadict = generated_queries, doc_class=doc_class)

    train_dataset = ConcatDataset([natural_queries, gen_queries])
    val_dataset = val_queries

    if args.base_data_dir_new or args.test_only:
        natural_queries_new = get_dataset(tokenizer=tokenizer, datadict = train_data_new, doc_class=doc_class)
        val_queries_new = get_dataset(tokenizer=tokenizer, datadict = val_data_new, doc_class=doc_class)
        gen_queries_new = get_dataset(tokenizer=tokenizer, datadict = generated_queries_new, doc_class=doc_class)

        train_dataset = ConcatDataset([train_dataset, natural_queries_new, gen_queries_new])
        val_dataset_new = val_queries_new

    if args.test_only:
        test_dataset = get_dataset(tokenizer=tokenizer, datadict = test_data, doc_class=doc_class)
        test_dataset_new = get_dataset(tokenizer=tokenizer, datadict = test_data_new, doc_class=doc_class)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.9, end_factor=1, total_iters=10)

    length = len(train_data) + len(generated_queries)
    if args.base_data_dir_new or args.test_only:
        length += len(train_data_new) + len(generated_queries_new)

    logger.info(f'dataset size:, {length}')

    val_length = len(val_data)

    logger.info(f'val_ dataset size:, {val_length}')

    if args.base_data_dir_new or args.test_only:
        val_length_new = len(val_data_new)
        logger.info(f'val_ new dataset size:, {val_length_new}')

    if args.test_only:
        test_length = len(test_data)
        test_length_new = len(test_data_new)
        logger.info(f'test dataset size:, {test_length}')
        logger.info(f'test new dataset size:, {test_length_new}')

    train_dataloader = get_dataloader(train_dataset, args.batch_size, tokenizer, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, args.batch_size, tokenizer)

    if args.base_data_dir_new or args.test_only:
        val_dataloader_new = get_dataloader(val_dataset_new, args.batch_size, tokenizer)

    if args.test_only:
        test_dataloader = get_dataloader(test_dataset, args.batch_size, tokenizer)
        test_dataloader_new = get_dataloader(test_dataset_new, args.batch_size, tokenizer)

    if args.test_only:
        logger.info('Test accuracy on the old dataset')
        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, test_dataloader)
        log_metrics(hit_at_1, hit_at_5, hit_at_10, mrr_at_10, test_length)

        logger.info('Test accuracy on the new dataset')
        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, test_dataloader_new)
        log_metrics(hit_at_1, hit_at_5, hit_at_10, mrr_at_10, test_length_new, ext='_new')

    
    logger.info(f'Validation accuracy on the old dataset:')
    hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader)
    log_metrics(hit_at_1, hit_at_5, hit_at_10, mrr_at_10, val_length)

    if args.base_data_dir_new:
        logger.info(f'Validation accuracy on the new dataset:')
        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader_new)
        log_metrics(hit_at_1, hit_at_5, hit_at_10, mrr_at_10, val_length_new, ext='_new')

    if args.test_only:
        return

    if not args.validate_only:
        if args.wandb_name:
            wandb.log({'Hits@1': hit_at_1/val_length, 'Hits@5': hit_at_5/val_length, \
                    'Hits@10': hit_at_10/val_length, 'MRR@10': mrr_at_10/val_length, "epoch": 0})
            if args.base_data_dir_new:
                wandb.log({'Hits@1_new': hit_at_1/val_length_new, 'Hits@5_new': hit_at_5/val_length_new, \
                        'Hits@10_new': hit_at_10/val_length_new, 'MRR@10_new': mrr_at_10/val_length_new, "epoch": 0})

        # Training
        for i in range(args.train_epochs):
            logger.info(f"Epoch: {i+1}")
            print(f"Learning Rate: {scheduler.get_last_lr()}")
            train(args, model, train_dataloader, optimizer, length)

            scheduler.step()
            hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader)

            logger.info(f'Epoch: {i+1}')
            logger.info(f'Evaluating on the old dataset')
            log_metrics(hit_at_1, hit_at_5, hit_at_10, mrr_at_10, val_length, args.wandb_name, i+1)

            if args.base_data_dir_new:
                hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader_new)

                logger.info(f'Evaluating on the new dataset')
                log_metrics(hit_at_1, hit_at_5, hit_at_10, mrr_at_10, val_length_new, args.wandb_name, i+1, ext='_new')

            cp = save_checkpoint(args.output_dir, model, args.output_name, i+1)
            logger.info('Saved checkpoint at %s', cp)
                                               

if __name__ == "__main__":
    main()


