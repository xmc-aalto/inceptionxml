# cluster from AttentionXML
import os
import tqdm
import joblib
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer

def get_sparse_feature(feature_file, label_file):
    sparse_x, _ = load_svmlight_file(feature_file, multilabel=True)
    sparse_labels = [i.replace('\n', '').split() for i in open(label_file)]
    return normalize(sparse_x), np.array(sparse_labels)

def build_tree_by_level(sparse_data_x, sparse_data_y, eps: float, max_leaf: int, levels: list, groups_path):
    print('Clustering')
    sparse_x, sparse_labels = get_sparse_feature(sparse_data_x, sparse_data_y)
    mlb = MultiLabelBinarizer(sparse_output=True)
    sparse_y = mlb.fit_transform(sparse_labels)
    joblib.dump(mlb, groups_path+'mlb')
    print('Getting Labels Feature')
    labels_f = normalize(sparse_y.T @ csc_matrix(sparse_x))
    print(F'Start Clustering {levels}')
    levels, q = [2**x for x in levels], None
    for i in range(len(levels)-1, -1, -1):
        if os.path.exists(F'{groups_path}-Level-{i}.npy'):
            print(F'{groups_path}-Level-{i}.npy')
            labels_list = np.load(F'{groups_path}-Level-{i}.npy', allow_pickle=True)
            q = [(labels_i, labels_f[labels_i]) for labels_i in labels_list]
            break
    if q is None:
        q = [(np.arange(labels_f.shape[0]), labels_f)]
    
    num_split = len([1 for node_i,_ in q if len(node_i) > max_leaf])
    while num_split:
        labels_list = np.asarray([x[0] for x in q])
        assert sum(len(labels) for labels in labels_list) == labels_f.shape[0]
        if len(labels_list) in levels:
            level = levels.index(len(labels_list))
            print(F'Finish Clustering Level-{level}')
            np.save(F'{groups_path}-Level-{level}.npy', np.asarray(labels_list))
        else:
            # print(F'Finish Clustering {len(labels_list)}')
            pass
        next_q = []
        max_size = max([len(node_i) for node_i, _ in q])
        print(f'Maximum size of node is {max_size}')
        for node_i, node_f in q:
            if len(node_i) > max_leaf:
                next_q += list(split_node(node_i, node_f, eps))
            else:
                # will always be balnced, so pass will also work
                next_q += [(node_i, node_f)]
                # np.save(F'{groups_path}-last.npy', np.asarray(labels_list)) # why save so many times??
        q = next_q
        print(f'Size of next_q {len(q)}')
        num_split = len([1 for node_i,_ in q if len(node_i) > max_leaf])
        print(f'Number of nodes to split is {num_split}')
        print()
    labels_list = np.asarray([x[0] for x in q])
    np.save(F'{groups_path}-last.npy', np.asarray(labels_list))
    
    print('Finish Clustering')
    return mlb


def split_node(labels_i: np.ndarray, labels_f: csr_matrix, eps: float):
    n = len(labels_i)
    c1, c2 = np.random.choice(np.arange(n), 2, replace=False)
    centers, old_dis, new_dis = labels_f[[c1, c2]].toarray(), -10000.0, -1.0
    l_labels_i, r_labels_i = None, None
    while new_dis - old_dis >= eps:
        dis = labels_f @ centers.T  # N, 2
        partition = np.argsort(dis[:, 1] - dis[:, 0])
        l_labels_i, r_labels_i = partition[:n//2], partition[n//2:]
        old_dis, new_dis = new_dis, (dis[l_labels_i, 0].sum() + dis[r_labels_i, 1].sum()) / n
        centers = normalize(np.asarray([np.squeeze(np.asarray(labels_f[l_labels_i].sum(axis=0))),
                                        np.squeeze(np.asarray(labels_f[r_labels_i].sum(axis=0)))]))
    return (labels_i[l_labels_i], labels_f[l_labels_i]), (labels_i[r_labels_i], labels_f[r_labels_i])

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=False, default='AmazonTitles-670K')
parser.add_argument('--tree', action='store_true')
parser.add_argument('--id', type=str, required=False, default='0')

args = parser.parse_args()

if __name__ == '__main__':
    dataset = args.dataset
    datapath = os.path.join('./Datasets/', dataset)
    
    if dataset == 'WikiSeeAlsoTitles-350K' or dataset == 'AmazonTitles-670K':
        final_name = f'label_group_{args.id}'
        final_name = os.path.join(datapath, final_name)
        train_file = os.path.join(datapath, 'bow-train.txt')
        labels_file = os.path.join(datapath, 'bow-labels.txt')
        
        mlb = build_tree_by_level(train_file, labels_file, 1e-4, 15, [], f'{final_name}')

        groups = np.load(f'{final_name}-last.npy', allow_pickle=True)
        new_group = []
        for group in groups:
            new_group.append([mlb.classes_[i] for i in group])
        np.save(f'{final_name}.npy', np.array(new_group))

    elif dataset == 'WikiTitles-500K' or dataset == 'AmazonTitles-3M':
        final_name = f'label_group_{args.id}'
        final_name = os.path.join(datapath, final_name)
        train_file = os.path.join(datapath, 'bow-train.txt')
        labels_file = os.path.join(datapath, 'bow-labels.txt')
        
        if dataset == 'WikiTitles-500K':
            mlb = build_tree_by_level(train_file, labels_file, 1e-4, 10, [], f'{final_name}')
        else:
            mlb = build_tree_by_level(train_file, labels_file, 1e-4, 30, [], f'{final_name}')            

        groups = np.load(f'{final_name}-last.npy', allow_pickle=True)
        new_group = []
        for group in groups:
            new_group.append([mlb.classes_[i] for i in group])
        np.save(f'{final_name}.npy', np.array(new_group))