import numpy as np
import torch
import torch.sparse
import torch.nn as nn
from torch.utils.data import Dataset
import tqdm
from xclib.data import data_utils as du
import os
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from xclib.utils.graph import normalize_graph
from eclare_tree import build_tree as EclareTree
from decaf_tree import build_tree as DecafTree
# from tree import build_tree as EclareTree 
from random_walks import PrunedWalk
from xclib.utils.sparse import retain_topk
import pickle as pkl

class ShortXMLData(Dataset):
    def __init__(self, X, y, params, use_clusters, mode='train'):
        assert mode in ["train", "eval", "test"]
        self.mode = mode
        self.X = torch.from_numpy(X)
        pre = 'trn' if mode == 'train' else 'tst'
        self.Y = du.read_sparse_file(os.path.join(params.data_path, f'{pre}_X_Y.txt'), dtype=np.float32)
        self.num_labels = params.num_labels

        self.label_to_cluster_ids = None
        self.label_space = torch.zeros(self.num_labels)

        if use_clusters:
            # clusters = np.load(os.path.join(params.data_path, params.cluster_name), allow_pickle=True)
            # clusters = [list(map(int, group)) for group in clusters]
            # self._init_clusters_(clusters)
            if mode == 'train':
                features = du.read_sparse_file(os.path.join(params.data_path, f'trn_X_Xf.txt'), dtype=np.float32)
                self.TreeX = normalize(features, norm='l2')
            # self.label_graph = self.load_graph(params)
            if 'eclare' in params.cluster_name:
                self.tree = EclareTree(b_factors=params.b_factors, method=params.cluster_method, 
                                    leaf_size=params.num_labels, force_shallow=True)
                self.build_eclare(params)
            elif 'decaf' in params.cluster_name or 'astec' in params.cluster_name:
                self.tree = DecafTree(b_factors=params.b_factors, method='parabel', force_shallow=True)
                self.build_astec(params)
            else:
                raise ValueError("Clustering type not implemented")
    
    def _init_clusters_(self, label_clusters):
        self.label_clusters = label_clusters
        self.num_clusters = len(label_clusters)
        self.cluster_space = torch.zeros(self.num_clusters)
        self.label_to_cluster_ids = np.zeros(self.num_labels, dtype=np.int64) - 1
        for idx, labels in enumerate(self.label_clusters):
            self.label_to_cluster_ids[labels] = idx

    def build_astec(self, params, label_repr=None):
        print(f"Loading clusters from {params.cluster_name}")
        if not os.path.exists(os.path.join(params.data_path, params.cluster_name)):
            label_repr = self.create_label_fts_vec() if label_repr is None else label_repr
            if 'decaf' in params.cluster_name:
                self.tree.fit(label_repr)
            elif 'astec' in params.cluster_name:
                self.tree.fit(label_repr, self.Y, params.dataset)
            self.tree.save(os.path.join(params.data_path, params.cluster_name))
        else:
            self.tree.load(os.path.join(params.data_path, params.cluster_name))
        clusters = self.tree._get_cluster_depth(0)
        self._init_clusters_(clusters)
        return clusters

    def create_label_fts_vec(self):
        _labels = self.Y.transpose()
        _features = self.TreeX
        lbl_cnt = _labels.dot(_features).tocsr()
        lbl_cnt = retain_topk(lbl_cnt, k=1000)
        return lbl_cnt
    
    def load_graph(self, params, word_embeds=None):
        print(os.path.join(params.data_path, params.graph_name))
        if not os.path.exists(os.path.join(
                params.data_path, params.graph_name)):
            # trn_y = du.read_sparse_file(os.path.join(params.data_path, 'trn_X_Y.txt'))
            # valid_lbs = np.loadtxt(params.label_indices, dtype=int)
            # trn_y = trn_y.tocsc()[:, valid_lbs]
            trn_y = self.Y
            n_lbs = trn_y.shape[1]
            diag = np.ones(n_lbs, dtype=np.int)

            if params.verbose_lbs > 0:
                verbose_labels = np.where(
                    np.ravel(trn_y.sum(axis=0) > params.verbose_lbs))[0]
                print("Verbose_labels:", verbose_labels.size)
                diag[verbose_labels] = 0
            else:
                verbose_labels = np.asarray([])
            diag = sp.diags(diag, shape=(n_lbs, n_lbs))
            print("Avg: labels", trn_y.nnz/trn_y.shape[0])
            trn_y = trn_y.dot(diag).tocsr()
            trn_y.eliminate_zeros()
            yf = None
            if word_embeds is not None:
                print("Using label features for PrunedWalk")
                emb = word_embeds.detach().cpu().numpy()
                yf = normalize(self.Yf.dot(emb)[:-1])
            graph = PrunedWalk(trn_y, yf=yf).simulate(
                params.walk_len, params.p_reset,
                params.graph_topk, max_dist=params.prune_max_dist)
            if verbose_labels.size > 0:
                graph = graph.tolil()
                graph[verbose_labels, verbose_labels] = 1
                graph = graph.tocsr()
            sp.save_npz(os.path.join(
                params.data_path, params.graph_name), graph)
        else:
            graph = sp.load_npz(os.path.join(
                params.data_path, params.graph_name))
        return graph

    def build_eclare(self, params, label_repr=None, word_embeds=None):
        print(f"Loading clusters from {params.cluster_name}")
        if not os.path.exists(os.path.join(params.data_path, params.cluster_name)):
            lbl_cnt = self.create_label_fts_vec() if label_repr is None else label_repr
            freq_y = np.ravel(self.Y.sum(axis=0))
            
            verb_lbs, norm_lbs = np.asarray([]), np.arange(freq_y.size)
            if params.verbose_lbs > 0:
                verb_lbs = np.where(freq_y > params.verbose_lbs)[0]
                norm_lbs = np.where(freq_y <= params.verbose_lbs)[0]

            label_graph = self.load_graph(params, word_embeds)
            n_gph = normalize_graph(label_graph)

            if params.cluster_method == 'AugParabel':
                print("Augmenting graphs")
                if isinstance(lbl_cnt, np.ndarray):
                    print("Using Dense Features")
                    lbl_cnt = n_gph.dot(normalize(lbl_cnt))
                elif sp.issparse(lbl_cnt):
                    print("Using Sparse Features")
                    print("Avg features", lbl_cnt.nnz / lbl_cnt.shape[0])
                    lbl_cnt = n_gph.dot(lbl_cnt).tocsr()
                    lbl_cnt = retain_topk(lbl_cnt.tocsr(), k=1000).tocsr()
                    print("Avg features", lbl_cnt.nnz / lbl_cnt.shape[0])
                else:
                    print("Do not understand the type")
                    exit(0)
            self.tree.fit(norm_lbs, verb_lbs, lbl_cnt)
            self.tree.save(os.path.join(params.data_path, params.cluster_name))
        else:
            self.tree.load(os.path.join(params.data_path, params.cluster_name))
        clusters = self.tree._get_cluster_depth(0)
        self._init_clusters_(clusters)
        return clusters

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        labels = torch.LongTensor(self.Y[idx].indices)           
        
        if self.label_to_cluster_ids is not None: 
            if self.mode=='train':            
                cluster_ids = np.unique(self.label_to_cluster_ids[labels])
                if cluster_ids[0] == -1:
                    cluster_ids = cluster_ids[1:]
                cluster_binarized = self.cluster_space.scatter(0, torch.tensor(cluster_ids), 1.0)
                return self.X[idx], labels, cluster_binarized
            else:
                # new_x = self.X[idx].copy()
                # try:
                #     zind = np.where(new_x == 0)[0][0]
                #     new_x[:zind] = np.random.permutation(new_x[:zind])
                # except:
                #     new_x = np.random.permutation(new_x)
                # return torch.LongTensor(new_x), labels
                return self.X[idx], labels
        else:
            label_binarized = self.label_space.scatter(0, labels, 1.0)
            return self.X[idx], label_binarized

# class WordRepDataset(ShortXMLData):
#     def __init__(self, X, y, params, use_clusters, mode='train'):
#         super().__init__(X, y, params, use_clusters, mode)
#         with open('aug_tok.pkl', 'rb') as f:
#             self.aug_data = pkl.load(f)
#         self.aug_prob = params.perturb_prob

#     def __getitem__(self, idx):
#         batch = super().__getitem__(idx)
#         if self.aug_data[idx] != '-':        
# 	    prob = torch.rand([])
#             if prob <= self.aug_prob:
#                 ch = random.choice(range(len(self.aug_data[idx])))
#                 batch[0] = torch.LongTensor(self.aug_data[idx][ch])
#         return **batch
