import torch
import torch.nn as nn
import torch.nn.functional as F
from SimpleSelfAttention import SimpleSelfAttention

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

class InceptionXML(nn.Module):
    def __init__(self, params, embedding_weights, train_dataset=None):
        super(InceptionXML, self).__init__()
        
        emb_dim = embedding_weights.shape[1]
        filter_channels = params.num_filters
        self.loss_fn = nn.BCEWithLogitsLoss() 

        self.lookup = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_weights), freeze=False, padding_idx = 0
        )

        self.proj_layer = nn.Sequential(
                                nn.Conv1d(params.seq_len, filter_channels, 1, stride=1),
                                nn.GELU(),
                                nn.BatchNorm1d(filter_channels),
                                nn.Dropout(0.25)
        )
        
        conv_stride = 4
        self.conv_layers = nn.ModuleList()
        for fsz in params.filter_sizes:
            conv_n = nn.Conv1d(filter_channels, filter_channels, fsz, padding=2*(fsz//conv_stride - 1), 
                                         padding_mode='circular', stride=conv_stride)
            self.conv_layers.append(conv_n)

        self.post_conv_layer1 = nn.Sequential(
                                nn.GELU(),
                                nn.BatchNorm1d(3*filter_channels),
                                nn.Dropout(0.25)
        )

        # self.avg_pool = nn.AdaptiveAvgPool1d(emb_dim//4)
        # self.alpha = nn.Parameter(torch.tensor([0.]))
        # self.beta = nn.Parameter(torch.tensor([0.]))

        # self.post_conv_layer1 = nn.Sequential(
        #                         nn.GELU(),
        #                         nn.BatchNorm1d(3*filter_channels),
        #                         nn.Dropout(0.25),
        #                         nn.Conv1d(3*filter_channels, filter_channels, 1),
        #                         nn.GELU(),
        #                         nn.BatchNorm1d(filter_channels)
        # )
        # self.post_drop = nn.Dropout(0.25)

        self.conv_layer2 = nn.Sequential(
                                nn.Conv1d(3*filter_channels, filter_channels, 16, stride=conv_stride),
                                # nn.Conv1d(filter_channels, filter_channels, 16, stride=conv_stride),
                                nn.GELU(),
                                nn.BatchNorm1d(filter_channels)
        )
    
        self.ext_hidden = nn.Sequential(
                           nn.Dropout(0.25),
                           nn.Linear(params.hidden_dims, params.hidden_dims),
                           nn.GELU(),
                           nn.BatchNorm1d(params.hidden_dims)
        )

        drop_ext = {'WikiSeeAlsoTitles-350K': 0.5, 'AmazonTitles-670K': 0.5, 'WikiTitles-500K': 0.4}
        self.ext_classifier = nn.Sequential(
                            nn.Dropout(drop_ext[params.dataset]),
                            nn.Linear(params.hidden_dims, params.num_labels)
        )

        self.init_weights()

        self.attn = SelfAttentionModule(params.seq_len, emb_dim, drop = 0.25)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y_batch=None):
        h_sa = self.attn(self.lookup(x))
        h_p = self.proj_layer(h_sa)

        h_convs = []
        for conv_layer in self.conv_layers:
            h_convs.append(conv_layer(h_p))

        h1 = self.post_conv_layer1(torch.cat(h_convs, 1))
        # h1 = self.post_drop(self.alpha.sigmoid()*h1 + self.beta.sigmoid()*self.avg_pool(h_p)) #added

        h2 = torch.flatten(self.conv_layer2(h1), 1)

        h_o = self.ext_hidden(h2)
        logits = self.ext_classifier(h_o + h2)
        
        if y_batch is not None:
            return logits, self.loss_fn(logits, y_batch)
        else:
            return logits
