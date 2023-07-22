import sys
sys.path.append('./utils/')
sys.path.append('./data/')
sys.path.append('./models/')
sys.path.append('/optimizers/')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
import random
from glove_utils import *
from data_utils import *
from ShortXMLData import ShortXMLData #, WordRepDataset
import Runner_Plus, Runner, Runner_Scaled
from InceptionXML import InceptionXML
from InceptionPlus import InceptionPlus
# from InceptionPlus_shortlist import InceptionPlus
# from CondIncXML import CondIncXML
# from SqeIncXML import SqeIncXML

NUM_LABELS = {'AmazonTitles-670K': 670091, 'AmazonTitles-3M': 2812281, 'WikiSeeAlsoTitles-350K': 352072, 'WikiTitles-500K' : 501070, 'LF-AmazonTitles-131K': 131073}
NUM_CLUSTERS = {'AmazonTitles-670K': 65536, 'AmazonTitles-3M': 131072, 'WikiSeeAlsoTitles-350K': 32768, 'WikiTitles-500K' : 65536, 'LF-AmazonTitles-131K': 16384}

def seed_everything(seed=29):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def collate_func(batch):
    collated = []
    for i, b in enumerate(zip(*batch)):
        if i != 1:
            b = torch.stack(b)
        collated.append(b)
    return collated

def main(params):
    params.model_name += "_data-{}_sql-{}_emb-{}".format(params.dataset, params.seq_len, params.embed_type)

    print('Saving Model to: ' + params.model_name)
    params.model_name = os.path.join('../saved_models/', params.model_name)
    params.num_labels = NUM_LABELS[params.dataset]
    params.num_clusters = NUM_CLUSTERS[params.dataset]
    params.data_path = os.path.join(params.data_dir, params.dataset)

    x_train, y_train, x_test, y_test, inv_prop, emb_weights, group_y = create_data(params) if params.create_data else load_data(params)

    seed_everything(params.seed)
    print(f"Initialized seed to {params.seed}")
    
    if not os.path.exists(params.model_name):
        os.makedirs(params.model_name)

    if len(params.load_model):
        params.load_model = os.path.join(params.model_name, params.load_model)

    collate_fn = None if params.model == 'InceptionXML' else collate_func
    use_clusters = False if params.model == 'InceptionXML' else True
    tr_dataset = ShortXMLData if params.perturb_prob == -1 else WordRepDataset
    train_dataset = tr_dataset(x_train, y_train, params, use_clusters) 
    test_dataset = ShortXMLData(x_test, y_test, params, use_clusters, mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    model = globals()[params.model](params, emb_weights, train_dataset)

    print(model)
    print("%"*100)
    print(params)
    print("%"*100, '\n')

    if params.dataset == 'AmazonTitles-3M':
        print("Using scaled runner")
        runner = Runner_Scaled.Runner(train_dataloader, test_dataloader, inv_prop)
    elif params.model != 'InceptionXML':
        runner = Runner_Plus.Runner(train_dataloader, test_dataloader, inv_prop)
    else:
        runner = Runner.Runner(train_dataloader, test_dataloader, inv_prop)
    runner.train(model, params)    

if __name__ == '__main__':
    # ------------------------ Params -------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mn', dest='model_name', type=str, default='', help='model name')
    parser.add_argument('--lm', dest='load_model', type=str, default="", help='model to load')
    parser.add_argument('--test', action='store_true', help='Testing mode or training mode')
    
    parser.add_argument('--mb', dest='batch_size', type=int, default=128, help='Size of minibatch, changing might result in latent layer variance overflow')
    parser.add_argument('--lr', dest='lr', type=float, default=5e-3, help='Learning Rate')
    parser.add_argument('--ep', dest='num_epochs', type=int, default=45, help='Number of epochs for training')
    parser.add_argument('--meta-ep', dest='meta_epoch', type=int, default=10, help='Number of epochs for classic detach training')
    parser.add_argument('--meta_scale', type=int, default=1, help='Scaling factor for meta loss')

    parser.add_argument('--data_dir', type=str, default="./../../Datasets/", help='data directory')
    parser.add_argument('--ds', dest='dataset', type=str, default="AmazonTitles-670K", help='dataset name')

    parser.add_argument('--embed_type', type=str, default='glove')
    parser.add_argument('--hd', dest='hidden_dims', type=int, default=480, help='hidden layer dimension')
    parser.add_argument('--seq_len', type=int, default=32, help='max sequence length of a document')

    parser.add_argument('--topK', type=int, default = 800, help = 'Max clusters to be taken at test time')
    parser.add_argument('--train_topK', type=int, default = 20, help = 'Max clusters to be taken at train time')
    parser.add_argument('--test_topK', type=int, default = 500, help = 'Max clusters to be taken at test time')
    # parser.add_argument('--shortlist_size', type=int, default = 1000, help = 'Labels to consider for loss')
    parser.add_argument('--model', type=str, default='InceptionXML', choices=['InceptionXML', 'InceptionPlus'])#, 'CondIncXML', 'SqeIncXML'])
    
    parser.add_argument('--filter_sizes', default=[4, 8, 16], nargs='+', help='number of filter sizes (could be a list of integer)', type=int)
    parser.add_argument('--num_filters', help='number of filters (i.e. kernels) in CNN model', type=int, default=32)
    
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--create_data', action='store_true', help='Create Data or load data')
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--sparse', action='store_true', help='Make embedding layers sparse')

    #Parabel Cluster params
    parser.add_argument('--cluster_name', default='clusters_eclare.pkl')
    parser.add_argument('--b_factors', default=[16], type=int, nargs='+')
    parser.add_argument('--cluster_method', default='AugParabel')
    parser.add_argument('--verbose_lbs', type=int, default=500)

    #Graph params
    parser.add_argument('--graph_name', default='graph_sparse.npz')
    parser.add_argument('--prune_max_dist', type=float, default=1.0)
    parser.add_argument('--p_reset', type=float, default=0.8)
    parser.add_argument('--walk_len', type=int, default=400)
    parser.add_argument('--graph_topk', type=int, default=10)
    parser.add_argument('--perturb_prob', type=float, default=-1)
    
    params = parser.parse_args()
    main(params)
