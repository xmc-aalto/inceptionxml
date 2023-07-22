import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from SimpleSelfAttention import SimpleSelfAttention
from torch.nn.utils import spectral_norm, rnn
from xclib.utils.graph import normalize_graph
import scipy.sparse as sp

class SelfAttentionModule(nn.Module):
   def __init__(self, seq_len, emb_dim, drop = 0.25):
       super().__init__()
       self.dropout = nn.Dropout(drop)
       self.sa_x = SimpleSelfAttention(seq_len)
       self.sa_cross = SimpleSelfAttention(emb_dim)
    
   def forward(self, x):
       x = self.dropout(x)
       h_x = self.sa_x(x).permute(0, 2, 1)
       h_sa = self.sa_cross(h_x, x.permute(0,2,1)).permute(0, 2, 1)

       return h_sa

class InceptionPlus(nn.Module):
    def __init__(self, params, embedding_weights, train_dataset):
        super(InceptionPlus, self).__init__()

        self.params = params
        self.candidates_topk = params.topK
        self.dataset = params.dataset
        self.num_labels = params.num_labels
        self.label_clusters = train_dataset.label_clusters
        self.meta_scale = params.meta_scale
        # self.label_graph = train_dataset.label_graph

        self._init_label_clusters()
        # self._sparse_matrix(params)

        conv_stride = 4
        emb_dim = embedding_weights.shape[1]
        hidden_dims = params.hidden_dims
        seq_len = params.seq_len
        filter_channels = params.num_filters
        
        self.filter_sizes = params.filter_sizes
        self.lookup = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_weights), freeze=False, padding_idx=0, sparse = params.sparse
        )
        self.loss_fn = nn.BCEWithLogitsLoss() 
        self.conv_layers = nn.ModuleList()
        self.meta_epoch = params.meta_epoch

        drop_meta = {'WikiSeeAlsoTitles-350K': 0.5, 'AmazonTitles-670K': 0.3, 'AmazonTitles-3M': 0.3, 'WikiTitles-500K': 0.4}
        drop_ext = {'WikiSeeAlsoTitles-350K': 0.5, 'AmazonTitles-670K': 0.5, 'AmazonTitles-3M': 0.4, 'WikiTitles-500K': 0.4}

        self.conv_layer0 = nn.Sequential(
                                nn.Conv1d(seq_len, filter_channels, 1, stride=1),
                                nn.GELU(),
                                nn.BatchNorm1d(filter_channels)
        )
        
        for fsz in self.filter_sizes:
            conv_n = nn.Conv1d(filter_channels, filter_channels, fsz, padding=2*(fsz//conv_stride - 1), 
                                         padding_mode='circular', stride=conv_stride)
            self.conv_layers.append(conv_n)

        self.post_conv_layer1 = nn.Sequential(
                                nn.GELU(),
                                nn.BatchNorm1d(3*filter_channels),
                                nn.Dropout(0.25)
        )

        self.conv_layer2 = nn.Sequential(
                                nn.Conv1d(3*filter_channels, filter_channels, 16, stride=conv_stride),
                                nn.GELU(),
                                nn.BatchNorm1d(filter_channels)
        )

        feat_size = 480

        if self.dataset == 'AmazonTitles-3M':
            hidden_dims = emb_dim
            self.meta_avg_pool = nn.AdaptiveAvgPool1d(emb_dim)
            self.ext_avg_pool = nn.AdaptiveAvgPool1d(emb_dim)

        self.meta_hidden = nn.Sequential(
                           nn.Dropout(0.25),
                           spectral_norm(nn.Linear(feat_size, hidden_dims)),
                           nn.GELU()
        )

        self.ext_hidden = nn.Sequential(
                           nn.Dropout(0.25),
                           spectral_norm(nn.Linear(feat_size, hidden_dims)),
                           nn.GELU()
        )
        self.ext_drop = nn.Dropout(drop_ext[self.dataset])

        num_meta_labels = self.label_clusters.shape[0] if self.label_clusters is not None else self.num_labels
        print(f'Number of Meta-labels: {num_meta_labels}; top_k: {self.candidates_topk}; meta-epochs: {self.meta_epoch}')

        self.meta_classifiers = nn.Sequential(
                            nn.Dropout(drop_meta[self.dataset]),
                            nn.Linear(hidden_dims, num_meta_labels)
        )
        
        self.ext_classif_embed = nn.Embedding(self.num_labels+1, hidden_dims, padding_idx=self.num_labels, sparse = params.sparse)

        self.init_weights()

        self.attn = SelfAttentionModule(params.seq_len, emb_dim, drop = 0.25)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.ext_classif_embed.weight[:-1])
        self.ext_classif_embed.weight[-1].data.fill_(0)

    def _init_label_clusters(self):
        max_labels = max([len(g) for g in self.label_clusters])
        num_clusters = len(self.label_clusters)
        for i in range(num_clusters):
            if len(self.label_clusters[i]) < max_labels:
                self.label_clusters[i] = np.pad(self.label_clusters[i], 
                                                (0, max_labels-len(self.label_clusters[i])), 
                                                constant_values=self.num_labels)
        self.label_clusters = torch.LongTensor(np.array(self.label_clusters)).cuda()

    def _BSMat(self, mat):
        mat = mat.tocsr()
        mat.sort_indices()
        mat = mat.tocoo()
        values = torch.FloatTensor(mat.data)
        index = torch.LongTensor(np.vstack([mat.row, mat.col]))
        shape = torch.Size(mat.shape)
        mat = torch.sparse_coo_tensor(index, values, shape)
        mat._coalesced_(True)
        return mat

    def _padded(self, mat, new_shape):
        mat = mat.tocoo()
        mat = sp.csr_matrix((mat.data, (mat.row, mat.col)), shape=new_shape)
        mat.sort_indices()
        return mat

    def _sparse_matrix(self, params):
        sh = self.label_graph.shape
        self.label_graph = self._padded(normalize_graph(self.label_graph), (sh[0]+1, sh[1]+1)) 
        self.label_graph.sort_indices()
        self.label_graph = self._BSMat(self.label_graph.tocsr()).cuda()

    def get_candidates(self, group_logits, group_gd=None):
        group_logits = torch.sigmoid(group_logits.detach())
        TF_logits = group_logits.clone()
        if group_gd is not None:
            TF_logits += group_gd
        scores, indices = torch.topk(TF_logits, k=self.candidates_topk)
        if self.is_training:
            scores = group_logits[torch.arange(group_logits.shape[0]).view(-1,1).cuda(), indices]
        candidates = self.label_clusters[indices] 
        candidates_scores = torch.ones_like(candidates) * scores[...,None] 
        
        return indices, candidates.flatten(1), candidates_scores.flatten(1)
    
    def forward(self, x, extreme_labels=None, group_labels=None, ep=None):
        self.is_training = extreme_labels is not None
        
        h_sa = self.attn(self.lookup(x))
        h0 = self.conv_layer0(h_sa)

        h_list = []

        for i in range(len(self.filter_sizes)):
            h_n = self.conv_layers[i](h0)
            h_list.append(h_n)

        h1 = self.post_conv_layer1(torch.cat(h_list, 1))

        h2 = self.conv_layer2(h1)
        h_f = torch.flatten(h2, 1)
        
        h_m = self.meta_hidden(h_f)
        if hasattr(self, 'meta_avg_pool'):
            meta_logits = self.meta_classifiers(h_m + self.meta_avg_pool(h_f))
        else:
            meta_logits = self.meta_classifiers(h_m + h_f)

        if self.label_clusters is None:
            classif_logits = meta_logits
            if self.is_training:
                loss = self.loss_fn(classif_logits, extreme_labels)
                return classif_logits, loss
            else:
                return classif_logits

        groups, candidates, group_candidates_scores = self.get_candidates(meta_logits, group_gd=group_labels)

        label_binarized, new_cands, new_group_cands = [], [], []
        for i in range(x.shape[0]):
            idxs = torch.where(candidates[i] != self.num_labels)[0]
            new_cands.append(candidates[i][idxs])
            new_group_cands.append(group_candidates_scores[i][idxs])
            if self.is_training:
                ext = extreme_labels[i].cuda()
                lab_bin = (new_cands[-1][..., None] == ext).any(-1).float()
                label_binarized.append(lab_bin)
        
        if self.is_training:
            labels = rnn.pad_sequence(label_binarized, True, 0).cuda()
        candidates = rnn.pad_sequence(new_cands, True, self.num_labels)
        group_candidates_scores = rnn.pad_sequence(new_group_cands, True, 0.)

        if self.is_training and ep <= self.meta_epoch:
            h_f = h_f.detach()
        
        h_c = self.ext_hidden(h_f)
        if hasattr(self, 'ext_avg_pool'):
            h_c = self.ext_drop(h_c + self.ext_avg_pool(h_f)).unsqueeze(-1)
        else:
            h_c = self.ext_drop(h_c + h_f).unsqueeze(-1)

        embed_weights = self.ext_classif_embed(candidates)
        classif_logits = torch.bmm(embed_weights, h_c).squeeze(-1)
        
        candidates_scores = torch.where(classif_logits == 0., -np.inf, classif_logits.double()).float().sigmoid()
        
        if self.is_training:
            loss = self.loss_fn(classif_logits, labels) + self.meta_scale*self.loss_fn(meta_logits, group_labels)
            return loss, candidates_scores, meta_logits.sigmoid(), candidates
        else:
            return candidates, candidates_scores, candidates_scores * group_candidates_scores
