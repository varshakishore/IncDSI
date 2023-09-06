import argparse
import joblib
import torch
import numpy as np
from BertModel import QueryClassifier
from utils import *
import time
from torch.optim import LBFGS, SGD
from tqdm import tqdm
from functools import partial
from transformers import BertTokenizer
from datetime import datetime
import json
import plotly.graph_objects as go


# ax imports for hyperparameter optimization
from ax.plot.contour import plot_contour
from ax.plot.slice import plot_slice
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.service.utils.report_utils import exp_to_df


from train_initial_model import validate_script
from utils import ArmijoSGD

def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def initialize_model(model_path, classifier_matrix):
    model = QueryClassifier(classifier_matrix.shape[0])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='cache')

    load_saved_weights(model, model_path)

    model.classifier.weight.data = classifier_matrix.detach().to('cpu')
    model = model.to('cuda')

    return model, tokenizer

def initialize_nq320k(data_dir,
                train_q,
                num_qs,
                embeddings_path, 
                model_path,
                multiple_queries=False,
                min_old_q=False,
                tune=False):
    set_seed()

    old_docs_list = joblib.load(os.path.join(data_dir, 'old_docs', 'doc_list.pkl'))
    class_num = len(old_docs_list)
    
    model = QueryClassifier(class_num)
    load_saved_weights(model, model_path)
    classifier_layer = model.classifier.weight.data
    
    assert multiple_queries, 'Must use multiple queries'
    
    # Sentence embeddings for generated queries
    old_gen_q_embeddings = joblib.load(os.path.join(embeddings_path, 'old-gen-embeddings.pkl')).to(classifier_layer.device)
    # Document ids for generated queries
    old_gen_q_doc_ids = joblib.load(os.path.join(embeddings_path, 'old-gen-docids.pkl')).to(classifier_layer.device)

    # Sentence embeddings for natural queries
    old_q_embeddings = joblib.load(os.path.join(embeddings_path, 'old-train-embeddings.pkl')).to(classifier_layer.device)
    # Document ids for generated queries
    old_q_doc_ids = joblib.load(os.path.join(embeddings_path, 'old-train-docids.pkl')).to(classifier_layer.device)

    old_qeries = torch.zeros(len(old_docs_list), 768).to(classifier_layer.device)
    if min_old_q:
        for i in tqdm(range(len(old_docs_list)), desc='Selecting min of old queries'):
            # Extract generated queries for the document
            gen_q_embs = old_gen_q_embeddings[old_gen_q_doc_ids == old_docs_list[i]][:num_qs]  # (num_qs, 768)
            # Extract natural queries for the document
            q_embs = old_q_embeddings[old_q_doc_ids == old_docs_list[i]]  # (*, 768)
            # Concatenate the two query embeddings
            q_embs = torch.cat([gen_q_embs, q_embs], dim=0)  # (*, 768)
            # Compute scores for each query
            doc_scores = torch.matmul(q_embs, classifier_layer[i])    # (num_qs)
            # Select the query with the lowest score
            min_idx = doc_scores.argmin()
            # Use the selected query embedding
            old_qeries[i] = q_embs[min_idx]
    else:
        for i in tqdm(range(len(old_docs_list)), desc='Selecting mean of old queries'):
            # Extract generated queries for the document
            gen_q_embs = old_gen_q_embeddings[old_gen_q_doc_ids == old_docs_list[i]][:num_qs]  # (num_qs, 768)
            # Extract natural queries for the document
            q_embs = old_q_embeddings[old_q_doc_ids == old_docs_list[i]]  # (*, 768)
            # Concatenate the two query embeddings
            q_embs = torch.cat([gen_q_embs, q_embs], dim=0)  # (*, 768)
            # Use the mean of the query embeddings
            old_qeries[i] = torch.mean(q_embs, dim=0)

    if tune:
        doc_split = 'tune'
        if 'MSMARCO' in data_dir:
            num_docs = 1000
    else:
        doc_split = 'new'
        if 'MSMARCO' in data_dir:
            num_docs = 10000
    
    new_docs_list = joblib.load(os.path.join(data_dir, f'{doc_split}_docs', 'doc_list.pkl'))
    if 'MSMARCO' in data_dir:
        new_docs_list = new_docs_list[:num_docs]
    new_gen_q_embeddings = joblib.load(os.path.join(embeddings_path, f'{doc_split}-gen-embeddings.pkl'))
    new_gen_q_doc_ids = joblib.load(os.path.join(embeddings_path, f'{doc_split}-gen-docids.pkl'))

    if train_q:
        print('using train set queries...')
        train_qs = joblib.load(os.path.join(embeddings_path, f'{doc_split}-train-embeddings.pkl')).to(classifier_layer.device)
        train_qs_doc_ids = joblib.load(os.path.join(embeddings_path, f'{doc_split}-train-docids.pkl')).to(classifier_layer.device)
    else: 
        train_qs = None
        train_qs_doc_ids = None

    return old_docs_list, new_docs_list, train_qs, train_qs_doc_ids, old_qeries, new_gen_q_embeddings, new_gen_q_doc_ids, classifier_layer, old_q_embeddings, old_q_doc_ids, old_gen_q_embeddings, old_gen_q_doc_ids

def add_noise(x, scale):
    return x + torch.randn(x.shape[0],x.shape[1]).to('cuda') * torch.norm(x, dim=1)[:, None] * scale

def addDocs(args, args_valid=None, ax_params=None):
    global time
    global start
    timelist = []
    failed_docs = []
    
    if args.dataset in ['nq320k', 'msmarco']:
        tune = (ax_params is not None) and (not args.tune_on_new)
        old_docs_list, new_docs_list, train_qs, train_qs_doc_ids, queries, new_gen_q_embeddings, new_gen_q_doc_ids, classifier_layer, old_q_embeddings, old_q_doc_ids, old_gen_q_embeddings, old_gen_q_doc_ids = initialize_nq320k(args.data_dir, args.train_q, args.num_qs, args.embeddings_path, args.model_path ,args.multiple_queries, args.min_old_q, tune)
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')
    
    num_new_docs = len(new_docs_list)

    if ax_params:
        lr = ax_params['lr']; lam = ax_params['lambda']; m1 = ax_params['m1']; m2 = ax_params['m2']; noise_scale = ax_params['noise_scale']; l2_reg = ax_params['l2_reg']
        print("Using hyperparameters:")
        print(ax_params)
    else:
        lr = args.lr; lam = args.lam; m1 = args.m1; m2 = args.m2; noise_scale = args.noise_scale; l2_reg = args.l2_reg

    # counter to keep track of number of added documents
    added_counter = len(classifier_layer)
    num_old_docs = len(classifier_layer)
    embedding_size = classifier_layer.shape[1]

    # add rows for the new docs
    classifier_layer = torch.cat((classifier_layer, torch.zeros(num_new_docs, embedding_size).to(classifier_layer.device)))
    queries = torch.cat((queries, torch.zeros(num_new_docs, embedding_size, device=queries.device)))

    for done, doc_id in enumerate(tqdm(new_docs_list, desc='Adding documents')):
        # this set of hyperparameters is not working
        if len(timelist) == 50 and len(failed_docs) >= 25 and ax_params: 
            print("Bad hyperparameters, skipping...")
            print("Failed docs: ", len(failed_docs))
            print(ax_params)
            return 0.0 
        
        qs = new_gen_q_embeddings[new_gen_q_doc_ids == doc_id][:args.num_qs]
        qs = qs.to('cuda')
        if args.train_q:
            # initialize the train queries matrix
            train_q = train_qs[train_qs_doc_ids == doc_id]
            train_q = train_q.to('cuda')
            # use generated queries and train queries
            qs = torch.cat((qs, train_q))
        
        if args.mean_new_q:
            qs = torch.mean(qs, 0, keepdim=True)

        if args.init == 'random':
            x = torch.nn.Linear(embedding_size, 1).weight.data.squeeze()
        elif args.init == 'mean':
            x = torch.mean(classifier_layer[:added_counter],0).clone().detach()
        elif args.init == 'mean_dpr':
            x = torch.mean(qs, dim=0).clone().detach()
        elif args.init == 'max':
            x = classifier_layer[torch.argmax(torch.einsum('nd,md->n', classifier_layer[:added_counter], qs)).item()].clone().detach()        
        x = x.to('cuda')
        x.requires_grad = True
        if args.optimizer == 'lbfgs':
            optimizer = LBFGS([x], lr=lr, tolerance_change=args.lbfgs_tolerance, line_search_fn='strong_wolfe')
        elif args.optimizer == 'sgd':
            optimizer = SGD([x], lr=lr)
        elif args.optimizer == 'armijo_sgd': # SGD + line search
            optimizer = ArmijoSGD([x], lr=lr, c=0.5, tau=0.5)
        else:
            raise NotImplementedError
        classifier_layer = classifier_layer.to('cuda')     
        queries = queries.to('cuda') 
        # compute document score for each query
        prod_to_old = torch.einsum('nd,md->nm', classifier_layer[:added_counter], qs)
        max_vals = torch.max(prod_to_old, dim=0).values

        # prepare an original query for adding noise
        if args.add_noise:
            qs_orig = torch.clone(qs)

        start = time.time()
        # Compute logits for old document classes
        with torch.no_grad():
            old_logits = (classifier_layer[:added_counter]*queries[:added_counter]).sum(1)
        for i in range(args.lbfgs_iterations):
            if args.add_noise:
                qs = add_noise(qs_orig, noise_scale)
            x.requires_grad = True
            def closure():
                loss = 0
                first_loss_term = torch.nn.functional.relu((max_vals+m1) - torch.einsum('md,d->m', qs, x))
                if args.squared_hinge:
                    first_loss_term = first_loss_term**2
                loss += lam * torch.sum(first_loss_term)

                prod = torch.einsum('d,nd->n', x, queries[:added_counter]) - old_logits + m2
                if args.symmetric_loss:
                    loss += torch.max(torch.maximum(prod, torch.zeros(len(prod)).to('cuda')))
                else:
                    second_loss_term = torch.nn.functional.relu(prod)
                    if args.squared_hinge:
                        second_loss_term = second_loss_term**2
                    loss += (1-lam)*(second_loss_term).sum()
                    
                loss += l2_reg*torch.sum(x**2)

                optimizer.zero_grad()
                loss.backward()
                return loss
            x_prev = x.clone().detach()
            loss = optimizer.step(closure)
            
            # Check if any element of tensor x is NaN
            if torch.isnan(x).any() and ax_params is not None:
                print("NaN detected, skipping...")
                return 0.
            with torch.no_grad():
                delta_norm = torch.linalg.vector_norm(x-x_prev).item()
                if delta_norm < args.update_tolerance:
                    break
        
            if loss == 0: break
        if loss > 200 and ax_params is not None:
            # Bad hyperparams, return early
            return 0.

        timelist.append(time.time() - start)

        if done % 50 == 0:
            print(f'Done {done} in {time.time() - start} seconds; loss={loss}')
        
        # Condition only meaningful if l2_reg is disabled
        if loss != 0 and l2_reg == 0: failed_docs.append(doc_id)
                
        # add to classifier_layer and embeddings
        classifier_layer[added_counter] = x
        if args.min_old_q:
            idx2add = torch.argmax(torch.matmul(qs, x.unsqueeze(dim=1)))
            queries[added_counter] = qs[idx2add]
        else:
            queries[added_counter] = qs.mean(dim=0)
        classifier_layer = classifier_layer.detach()
        queries = queries.detach()
        loss = loss.detach()
        added_counter += 1

    if ax_params is not None:
        joblib.dump(classifier_layer, os.path.join(args.write_path_dir, 'temp.pkl'))
        model, tokenizer = initialize_model(args.model_path, classifier_layer)
        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate_script(args_valid, tokenizer, model, doc_type='new' if args.tune_on_new else 'tune', split='valid')
        print('Accuracy on new valid queries')
        print(hit_at_1, hit_at_5, hit_at_10, mrr_at_10)
        if args.bayesian_target == 'new_val':
            print(ax_params)
            print(f'New hits@1: {hit_at_1}')
            return hit_at_1.item()
        
        old_hit_at_1, old_hit_at_5, old_hit_at_10, old_mrr_at_10 = validate_script(args_valid, tokenizer, model, doc_type='old', split='valid')
        print('Accuracy on old valid queries')
        print(old_hit_at_1, old_hit_at_5, old_hit_at_10, old_mrr_at_10)
        
        beta = args.harmonic_beta
        harmonic_mean = (1+beta**2)*(mrr_at_10*old_mrr_at_10)/((beta**2)*mrr_at_10 + old_mrr_at_10)
        print(ax_params)
        print(f'Old MRR@10: {old_mrr_at_10}')
        print(f'New MRR@10: {mrr_at_10}')
        print(f'harmonic_mean MRR@10: {harmonic_mean}')
        assert args.bayesian_target == 'harmonic_mean'
        
        return harmonic_mean.item()
        
    return failed_docs, classifier_layer, queries, np.asarray(timelist).mean(), timelist

def validate_on_splits(val_dir, model_path, data_dir, write_path_dir=None, permutation_seed=None):
    eval_doc_type = 'new'
    classifier_layer_path = os.path.join(val_dir, 'classifier_layer.pkl')
    args_valid = get_validation_arguments(model_path, classifier_layer_path, data_dir)
    classifier_layer = joblib.load(classifier_layer_path)
    model, tokenizer = initialize_model(model_path, classifier_layer)
    hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate_script(args_valid, tokenizer, model, doc_type=eval_doc_type, split='unseenq', permutation_seed=permutation_seed)
    print('Accuracy on new generated queries')
    print(hit_at_1, hit_at_5, hit_at_10, mrr_at_10)

    if write_path_dir is not None:
        with open(os.path.join(write_path_dir, 'log.txt'), 'a') as f:
            f.write('\n')
            f.write(f'Accuracy on {eval_doc_type} generated queries: \n')
            f.write(f'hit_at_1: {hit_at_1}\n')
            f.write(f'hit_at_5: {hit_at_5}\n')
            f.write(f'hit_at_10: {hit_at_10}\n')
            f.write(f'mrr_at_10: {mrr_at_10}\n')


    for split in ['valid', 'test']:
        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate_script(args_valid, tokenizer, model, doc_type='old', split=split, permutation_seed=permutation_seed)
        print(f'Accuracy on old {split} queries')
        print(hit_at_1, hit_at_5, hit_at_10, mrr_at_10)

        if write_path_dir is not None:
            with open(os.path.join(write_path_dir, 'log.txt'), 'a') as f:
                f.write('\n')
                f.write(f'Accuracy on old {split} queries: \n')
                f.write(f'hit_at_1: {hit_at_1}\n')
                f.write(f'hit_at_5: {hit_at_5}\n')
                f.write(f'hit_at_10: {hit_at_10}\n')
                f.write(f'mrr_at_10: {mrr_at_10}\n')

        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate_script(args_valid, tokenizer, model, doc_type=eval_doc_type, split=split, permutation_seed=permutation_seed)
        print(f'Accuracy on {eval_doc_type} {split} queries')
        print(hit_at_1, hit_at_5, hit_at_10, mrr_at_10)

        if write_path_dir is not None:
            with open(os.path.join(write_path_dir, 'log.txt'), 'a') as f:
                f.write('\n')
                f.write(f'Accuracy on {eval_doc_type} {split} queries: \n')
                f.write(f'hit_at_1: {hit_at_1}\n')
                f.write(f'hit_at_5: {hit_at_5}\n')
                f.write(f'hit_at_10: {hit_at_10}\n')
                f.write(f'mrr_at_10: {mrr_at_10}\n')

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=0.008, type=float, help="initial learning rate for optimization")
    parser.add_argument("--lam", default=6, type=float, help="lambda for optimization")
    parser.add_argument("--m1", default=0.03, type=float, help="margin for constraint 1")
    parser.add_argument("--m2", default=0.03, type=float, help="margin for constraint 2")
    parser.add_argument("--l2_reg", default=0.0, type=float, help="l2 regularization for the weights")
    parser.add_argument(
        "--dataset", 
        default='nq320k', 
        choices=['nq320k', 'msmarco'], 
        help='which dataset to use')
    parser.add_argument(
        "--optimizer", 
        default='sgd', 
        choices=['sgd', 'armijo_sgd', 'lbfgs'], 
        help='which optimizer to use')
    parser.add_argument("--squared_hinge", action="store_true", help="square the hinge loss (speeds up optimization)")
    parser.add_argument(
        "--bayesian_target", 
        default='new_val', 
        choices=['new_val', 'harmonic_mean'], 
        help='Target metric for hyperparameter tuning')
    parser.add_argument("--harmonic_beta", default=1, type=int, help="beta for harmonic mean")
    parser.add_argument("--lbfgs_tolerance", default=1e-3, type=float, help="tolerance for lbfgs",)
    parser.add_argument("--update_tolerance", default=1e-3, type=float, help="tolerance for the norm of the update",)
    parser.add_argument("--num_new_docs", default=None, type=int, help="number of new documents to add")
    parser.add_argument("--lbfgs_iterations", default=1000, type=int, help="number of iterations for lbfgs")
    parser.add_argument("--trials", default=30, type=int, help="number of trials to run for hyperparameter tuning")
    parser.add_argument("--write_path_dir", default=None, type=str, help="path to write classifier layer to")
    parser.add_argument("--tune_parameters", action="store_true", help="flag for tune parameters")
    parser.add_argument("--tune_on_new", action="store_true", help="flag for tune parameters")
    parser.add_argument("--multiple_queries", action="store_true", help="flag for multiple_queries")
    parser.add_argument("--num_qs", default=10, type=int, help="number of generated queries to use")
    parser.add_argument("--train_q", action="store_true", help="if we are using train queries to add documents")
    parser.add_argument("--add_noise", action="store_true", help="add noise to query embeddings when adding document")
    parser.add_argument("--add_noise_w_margin", action="store_true", help="add noise and keep the margin")
    parser.add_argument("--noise_scale", default="0.001", type=float, help="how much noise to add to the query embeddings")
    parser.add_argument("--symmetric_loss", action="store_true", help="loss from the two terms are symmetric")
    parser.add_argument("--min_old_q", action="store_true", help="uses the old query with the tightest constraint")
    parser.add_argument("--mean_new_q", action="store_true", help="uses the old query with the tightest constraint")
    parser.add_argument("--DSIplus", action="store_true", help="whether or not in the dsi++ setting")

    parser.add_argument("--permutation_seed", default=None, type=int, help="seed for permutation of new docs")

    parser.add_argument(
        "--init", 
        default='random', 
        choices=['random', 'mean', 'mean_dpr', 'max'], 
        help='way to initialize the classifier vector')
    parser.add_argument(
        "--data_dir", 
        default=None, 
        type=str, 
        help="path to data directory")
    parser.add_argument(
        "--embeddings_path", 
        default=None, 
        type=str, 
        help="path to embeddings")
    parser.add_argument(
        "--model_path", 
        default=None, 
        type=str, 
        help="path to model")
    parser.add_argument(
        "--train_q_doc_id_map_path", 
        default=None, 
        type=str, 
        help="path to doc_id to train_qs mapping")
    parser.add_argument(
        "--val",
        action="store_true",
        help="whether or not just to run the evaluation"
    )
    parser.add_argument(
        "--val_path",
        default=None,
        type=str,
        help="the folder for evaluation"
    )
    
    args = parser.parse_args()

    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.write_path_dir = args.write_path_dir if args.write_path_dir is not None else f'none/{datetime_str}'

    return args

def get_validation_arguments(model_path, optimized_embeddings_path, base_data_dir):
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
        default=None,
        type=int,
        help="Number of train epochs",
    )

    parser.add_argument(
        "--model_name",
        default='T5-base',
        choices=['T5-base', 'bert-base-uncased'],
        help="Model name",
    )

    parser.add_argument(
        "--base_data_dir",
        type=str,
        default="/home/vk352/dsi/data/NQ320k",
        help="where the train/test/val data is located",
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
        default=5e-4,
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
        "--initialize_embeddings",
        default=None,
        type=str,
        help="file for the embedding matrix",
    )

    parser.add_argument(
        "--optimized_embeddings",
        default=None,
        type=str,
        help="file for the optimized embedding matrix",
    )

    parser.add_argument(
        "--ance_embeddings",
        action="store_true",
        help="are these embeddings from ance",
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

    args = parser.parse_args(['--freeze_base_model', 
    '--output_dir', '/home/vk352/dsi/outputs/dpr5_finetune_0.001_filtered_fixed_new/', '--model_name', 'bert-base-uncased', 
    '--batch_size', '1600', 
    '--initialize_model', model_path,
    '--optimized_embeddings', optimized_embeddings_path,
    '--base_data_dir', base_data_dir,])

    return args

def exists(x):
    return x is not None

def set_file_paths(args):
    nq320k_filepaths = {'embeddings_path':'/home/vk352/IncDSI/test_output/saved_embeddings/',
                                'model_path':'/home/vk352/IncDSI/test_output/initial_model/base_model_epoch20',
                                'data_dir':'/home/vk352/dsi/data/NQ320k'
                                }
    msmarco_filepaths = {'embeddings_path':'/home/jl3353/dsi/msmarco_outputs/MSMARCO_2_bs1600lr5e-4/finetune_old_epoch10',
                                'model_path':'/home/cw862/DSI/dsi/outputs/MSMARCO_2_bs1600lr5e-4/finetune_old_epoch10',
                                'data_dir':'/home/cw862/MSMARCO/'
                                }
   
    filepath_defaults = {'nq320k':nq320k_filepaths, 'msmarco':msmarco_filepaths}

    arg_dict = vars(args)
    for k,v in filepath_defaults[args.dataset].items():
        arg_v = arg_dict[k]
        assert exists(v) or exists(arg_v), f'Need to define an argument or a default for {args.dataset}:{k}'
        if not exists(arg_v):
            arg_dict[k] = v


def save_bayesian_opt_plots(args, model, experiment):
    plot = plot_contour(model=model, param_x='m1', param_y='m2', metric_name='val_acc')
        
    data = plot[0]['data']
    lay = plot[0]['layout']

    fig = {
        "data": data,
        "layout": lay,
    }
    go.Figure(fig).write_image(os.path.join(args.write_path_dir, "margin_tradeoff.png"), format="png")

    plot = plot_slice(model, "l2_reg", "val_acc")
        
    data = plot[0]['data']
    lay = plot[0]['layout']

    fig = {
        "data": data,
        "layout": lay,
    }
    go.Figure(fig).write_image(os.path.join(args.write_path_dir, "l2_reg.png"), format="png")

    plot = plot_slice(model, "lambda", "val_acc")
        
    data = plot[0]['data']
    lay = plot[0]['layout']

    fig = {
        "data": data,
        "layout": lay,
    }
    go.Figure(fig).write_image(os.path.join(args.write_path_dir, "lambda.png"), format="png")

    best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
    plot = optimization_trace_single_method(
        y=np.maximum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="val_acc",
    )

    data = plot[0]['data']
    lay = plot[0]['layout']

    fig = {
        "data": data,
        "layout": lay,
    }
    go.Figure(fig).write_image(os.path.join(args.write_path_dir, "best_objective_plot.png"), format="png", width=1000, height=1000, scale=1)



def main():
    set_seed()
    args = get_arguments()
    set_file_paths(args)
    os.makedirs(args.write_path_dir, exist_ok=True)
    with open(os.path.join(args.write_path_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    if args.val:
        validate_on_splits(args.val_path, args.model_path, args.data_dir, args.val_path)
        return 

    if args.tune_parameters:
        print("Tuning parameters")
        os.makedirs(args.write_path_dir, exist_ok=True)
        args_valid = get_validation_arguments(args.model_path, os.path.join(args.write_path_dir, 'temp.pkl'), args.data_dir)

        if args.add_noise:
            best_parameters, values, experiment, model = optimize(
            parameters=[
                {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
                {"name": "lambda", "value_type": "float", "type": "range", "bounds": [.001, .999], "log_scale": False},
                {"name": "m1", "type": "fixed", "value": 0.0, "log_scale": False},
                {"name": "m2", "type": "fixed", "value": 0.0, "log_scale": False},
                {"name": "l2_reg", "type": "fixed", "value": 0.0, "log_scale": False },
                {"name": "noise_scale", "type": "range", "bounds": [1e-3, 1.0], "log_scale": True}
            ],
            evaluation_function=partial(addDocs, args, args_valid),
            objective_name='val_acc',
            total_trials=args.trials,
            minimize=False,
            )
        
        elif args.add_noise_w_margin:
            best_parameters, values, experiment, model = optimize(
            parameters=[
                {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
                {"name": "lambda", "value_type": "float", "type": "range", "bounds": [.001, .999], "log_scale": False},
                {"name": "m1", "type": "range", "bounds": [1e-5, 1.0], "log_scale": True},
                {"name": "m2", "type": "range", "bounds": [1e-5, 1.0], "log_scale": True},
                {"name": "l2_reg", "type": "fixed", "value": 0.0, "log_scale": False },
                {"name": "noise_scale", "type": "range", "bounds": [1e-3, 1.0], "log_scale": True}
            ],
            evaluation_function=partial(addDocs, args, args_valid),
            objective_name='val_acc',
            total_trials=args.trials,
            minimize=False,
            )

        elif args.symmetric_loss:
            best_parameters, values, experiment, model = optimize(
            parameters=[
                {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
                {"name": "lambda", "type": "fixed", "value": .5, "log_scale": True},
                {"name": "m1", "type": "range", "bounds": [1e-5, 1.0], "log_scale": True},
                {"name": "m2", "type": "range", "bounds": [1e-5, 1.0], "log_scale": True},
                {"name": "l2_reg", "type": "fixed", "value": 0.0, "log_scale": False },
                {"name": "noise_scale", "type": "fixed", "value": 0.0, "log_scale": False}
            ],
            evaluation_function=partial(addDocs, args, args_valid),
            objective_name='val_acc',
            total_trials=args.trials,
            minimize=False,
            )


        else:
            # ax optimize
            parameters = []
            if args.optimizer == 'lbfgs':
                parameters.append({"name": "lr", "value_type": "float", "type": "fixed", "value": 1, "log_scale": False})
            elif args.optimizer == 'sgd':
                parameters.append({"name": "lr", "value_type": "float", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True})
            else:
                raise NotImplementedError
            parameters += [
                {"name": "lambda", "value_type": "float", "type": "range", "bounds": [.05, .95], "log_scale": False},
                {"name": "m1", "value_type": "float", "type": "range", "bounds": [0.0, 8.0], "log_scale": False},
                {"name": "m2", "value_type": "float", "type": "range", "bounds": [0.0, 8.0], "log_scale": False},
                {"name": "l2_reg", "value_type": "float", "type": "range", "bounds": [1e-10, 1e-3], "log_scale": True},
                {"name": "noise_scale", "type": "fixed", "value": 0.0, "log_scale": False }
            ]
            with open(os.path.join(args.write_path_dir, 'tuning_config.json'), 'w') as f:
                json.dump(parameters, f, indent=2)

            best_parameters, values, experiment, model = optimize(
            parameters=parameters,
            evaluation_function=partial(addDocs, args, args_valid),
            objective_name='val_acc',
            total_trials=args.trials,
            minimize=False,
            )

        save_bayesian_opt_plots(args, model, experiment)

        print(exp_to_df(experiment)) 
        print(f'best_parameters')
        print(f'lr: {best_parameters["lr"]}')
        print(f'lambda: {best_parameters["lambda"]}')
        print(f'm1: {best_parameters["m1"]}')
        print(f'm2: {best_parameters["m2"]}')
        print(f'l2_reg: {best_parameters["l2_reg"]}')
        print(f'noise_scale: {best_parameters["noise_scale"]}')

        args.lr = best_parameters['lr']
        args.lam = best_parameters['lambda']
        args.m1 = best_parameters['m1']
        args.m2 = best_parameters['m2']
        args.l2_reg = best_parameters['l2_reg']
        args.noise_scale = best_parameters['noise_scale']

        if args.write_path_dir is not None:
            print("Writing to directory: ", args.write_path_dir)
            os.makedirs(args.write_path_dir, exist_ok=True)
            with open(os.path.join(args.write_path_dir, 'log.txt'), 'a') as f:
                f.write('\n')
                f.write(f'Target: {args.bayesian_target}\n')
                if args.bayesian_target == 'harmonic_mean':
                    f.write(f'Beta: {args.harmonic_beta}\n')
                f.write(f'Best Parameters:\n')
                f.write(f'lr: {args.lr}\n')
                f.write(f'lambda: {args.lam}\n')
                f.write(f'm1: {args.m1}\n')
                f.write(f'm2: {args.m2}\n')
                f.write(f'l2_reg: {args.l2_reg}\n')
                f.write(f'noise_scale: {args.noise_scale}\n')
                f.write('\n')
                f.write(f'experiment: {exp_to_df(experiment).to_csv()}\n')
                f.write(f'-'*100)
    
    print("Adding documents")
    failed_docs, classifier_layer, embeddings, avg_time, timelist = addDocs(args)


    if args.write_path_dir is not None:
        print("Writing to directory: ", args.write_path_dir)
        os.makedirs(args.write_path_dir, exist_ok=True)
        joblib.dump(classifier_layer, os.path.join(args.write_path_dir, 'classifier_layer.pkl'))
        joblib.dump(embeddings, os.path.join(args.write_path_dir, 'embeddings.pkl'))
        joblib.dump(failed_docs, os.path.join(args.write_path_dir, 'failed_docs.pkl'))
        joblib.dump(timelist, os.path.join(args.write_path_dir, 'timelist.pkl'))

        # save the final model
        classifier_layer_path = os.path.join(args.write_path_dir, 'classifier_layer.pkl')
        classifier_layer = joblib.load(classifier_layer_path)
        model, _ = initialize_model(args.model_path, classifier_layer)
        cp = save_checkpoint(args.write_path_dir, model, "final_model")

        with open(os.path.join(args.write_path_dir, 'log.txt'), 'a') as f:
            f.write('\n')
            f.write(f'Hyperparameters: opt={args.optimizer}, squared_hinge={args.squared_hinge}, num_qs={args.num_qs}, train_q={args.train_q}, min_old_q={args.min_old_q}, lr={args.lr},  m1={args.m1}, m2={args.m2}, lambda={args.lam}, l2_reg={args.l2_reg}, noise_scale={args.noise_scale}\n')
            f.write('\n')
            f.write(f'Num failed docs: {len(failed_docs)}\n')
            f.write(f'Final time: {np.asarray(timelist).sum()}\n')

        validate_on_splits(args.write_path_dir, args.model_path, args.data_dir, args.write_path_dir, args.permutation_seed)


if __name__ == "__main__":
    main()