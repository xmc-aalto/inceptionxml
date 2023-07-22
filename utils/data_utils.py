from glove_utils import *  
from torch.distributions import Beta
import torch
import torch.nn.functional as F

def create_data(args):
    print(f"Creating {args.embed_type} data")
    X_trn, Y_trn, X_tst, Y_tst, inv_prop, emb_weights = load_short_data(args)
    
    X_trn = X_trn.astype(np.int32)
    X_tst = X_tst.astype(np.int32)

    try:
        group_y = np.load(os.path.join(args.data_path, f'label_group_{args.num_clusters}.npy'), allow_pickle=True)
    except:
        group_y = None

    np.save(os.path.join(args.data_path, 'trn_labels'), Y_trn)
    np.save(os.path.join(args.data_path, 'tst_labels'), Y_tst)
    np.save(os.path.join(args.data_path, 'inv_prop.npy'), inv_prop)

    np.save(os.path.join(args.data_path, 'x_train'), X_trn)
    np.save(os.path.join(args.data_path, 'x_test'), X_tst)
    np.save(os.path.join(args.data_path, 'emb_weights'), emb_weights)

    return X_trn, Y_trn, X_tst, Y_tst, inv_prop, emb_weights, group_y


def load_data(args):
    print(f"Loading {args.embed_type} data")
    
    try:
        group_y = np.load(os.path.join(args.data_path, f'label_group_{args.num_clusters}.npy'), allow_pickle=True)
    except:
        group_y = None

    Y_trn = np.load(os.path.join(args.data_path, 'trn_labels.npy'), allow_pickle=True)
    Y_tst = np.load(os.path.join(args.data_path, 'tst_labels.npy'), allow_pickle=True)
    X_trn = np.load(os.path.join(args.data_path, args.embed_type, 'x_train.npy'))
    X_tst = np.load(os.path.join(args.data_path, args.embed_type, 'x_test.npy'))

    inv_prop = np.load(os.path.join(args.data_path, 'inv_prop.npy'))

    emb_weights = np.load(os.path.join(args.data_path, args.embed_type, 'emb_weights.npy')).astype(np.float32)
    # perm = np.random.permutation(300)
    # emb_weights = emb_weights[:, perm]
    # X_trn = np.where(X_trn == 66666, 0, X_trn)
    # X_tst = np.where(X_tst == 66666, 0, X_tst)

    return X_trn, Y_trn, X_tst, Y_tst, inv_prop, emb_weights, group_y

# def load_data(args):
#     print(f"Loading {args.embed_type} data")
    
#     try:
#         group_y = np.load(os.path.join(args.data_path, f'label_group_{args.num_clusters}.npy'), allow_pickle=True)
#         for i, cluster in enumerate(group_y):
#             group_y[i] = np.array(cluster, dtype=np.int32)
#     except:
#         print('Label clusters not found. Running without scaling up.')
#         group_y = None

#     Y_trn = np.load(os.path.join(args.data_path, 'trn_labels.npy'), allow_pickle=True)
#     Y_tst = np.load(os.path.join(args.data_path, 'tst_labels.npy'), allow_pickle=True)
#     X_trn = np.load(os.path.join(args.data_path, args.embed_type, 'x_train.npy'))
#     X_tst = np.load(os.path.join(args.data_path, args.embed_type, 'x_test.npy'))
#     X_lbl = np.load(os.path.join(args.data_path, args.embed_type, 'x_label.npy'))

#     inv_prop = np.load(os.path.join(args.data_path, 'inv_prop.npy'))
#     emb_weights = np.load(os.path.join(args.data_path, args.embed_type, 'emb_weights.npy'))
#     emb_weights[0] = np.zeros((1, emb_weights.shape[1]))
    
#     # data = (group_y, X_trn, X_tst, X_lbl, Y_trn, Y_tst, inv_prop, emb_weights)
#     data = (X_trn, Y_trn, X_tst, Y_tst, inv_prop, emb_weights, group_y)

#     # if args.embed_type == 'fasttext':
#     #     W_trn = np.load(os.path.join(args.data_path, args.embed_type, 'w_train.npy'))
#     #     W_tst = np.load(os.path.join(args.data_path, args.embed_type, 'w_test.npy'))
#     #     W_lbl = np.load(os.path.join(args.data_path, args.embed_type, 'w_label.npy'))
#     #     data = (group_y, (X_trn, W_trn), (X_tst, W_tst), (X_lbl, W_lbl), Y_trn, Y_tst, inv_prop, emb_weights)

#     return data
