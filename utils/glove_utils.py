import numpy as np
import os
import re
import scipy.sparse as sp
from itertools import chain
from collections import Counter
from nltk.corpus import stopwords
import json
from tqdm import tqdm
from w2v import load_word2vec
from scipy import sparse

cachedStopWords = stopwords.words("english")


def clean_str(string):
    string = re.sub(r"_", " ", string)
    string = ' '.join([word for word in string.split() if word not in cachedStopWords])
    string = re.sub('(?<=[A-Za-z]),', ' ', string)
    string = re.sub(r"(),!?", "", string)
    string = re.sub(r"[^A-Za-z0-9\.\'\`]", " ", string)
    string = re.sub('(?<=[A-Za-z])\.', '', string)
    string = re.sub(r'([\d]+)([A-Za-z]+)', '\g<1> \g<2>', string)
    string = re.sub(r"\'s ", " ", string)
    string = re.sub(r"s\' ", " ", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()
    return string.split()


def pad_sentences(sentences, padding_word="<PAD/>", max_length=500):
    sequence_length = min(max(len(x) for x in sentences), max_length)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < max_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:max_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def load_data_and_labels(data, labels, args):
    row_idx, col_idx, val_idx = [], [], []
    for i in range(len(labels)):
        l_list = labels[i]
        for y in l_list:
            row_idx.append(i)
            col_idx.append(y)
            val_idx.append(1)
    m = max(row_idx) + 1
    n = args.num_labels
    Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
    return x_text, Y


def build_vocab(sentences, vocab_size=50000):
    word_counts = Counter(chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # append <UNK/> symbol to the vocabulary
    vocabulary['<UNK/>'] = len(vocabulary)
    vocabulary_inv.append('<UNK/>')
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    x = np.array([np.array([vocabulary[word] if word in vocabulary else vocabulary['<UNK/>']
                 for word in sentence]) for sentence in sentences])
    #x = np.array([[vocabulary[word] if word in vocabulary else len(vocabulary) for word in sentence] for sentence in sentences])
    return x

def get_inv_prop(labels, args):
    print("Creating inv_prop file")
    
    A = {'AmazonTitles-670K': 0.6, 'AmazonTitles-3M': 0.6, 'WikiSeeAlsoTitles-350K': 0.55, 'WikiTitles-500K' : 0.5}
    B = {'AmazonTitles-670K': 2.6, 'AmazonTitles-3M': 2.6, 'WikiSeeAlsoTitles-350K': 1.5, 'WikiTitles-500K': 0.4}

    a, b = A[args.dataset], B[args.dataset]

    row_idx, col_idx, val_idx = [], [], []
    for i in range(len(labels)):
        l_list = labels[i]
        for y in l_list:
            row_idx.append(i)
            col_idx.append(y)
            val_idx.append(1)
    m = max(row_idx) + 1
    n = args.num_labels
    Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))

    num_labels = Y.shape[1]
    num_samples = Y.shape[0]
    inv_prop = np.array(Y.sum(axis=0)).ravel()
    
    c = (np.log(num_samples) - 1) * np.power(b+1, a)
    inv_prop = 1 + c * np.power(inv_prop + b, -a)
    
    return inv_prop

def data_cleaner(data):
    for i, t in enumerate(data):
        data[i] = clean_str(t)
        if len(data[i]) == 0:
            data[i] = t.split()

    return data

def load_short_data(args):
    
    VOCAB_SIZE = {'AmazonTitles-670K': 66666, 'AmazonTitles-3M': 165431, 'WikiSeeAlsoTitles-350K': 91414, 'WikiTitles-500K' : 185479}

    trn_data, trn_labels = [], []
    tst_data, tst_labels = [], []
    with open(os.path.join(args.data_path, 'trn.json')) as fin:
        for info in tqdm(fin.readlines(), desc='Reading training data'):
            info = json.loads(info)
            trn_data.append(info['title'])
            trn_labels.append(np.array(info['target_ind']))

    with open(os.path.join(args.data_path, 'tst.json'), 'r') as fin:
        for info in tqdm(fin.readlines(), desc='Reading testing data'):
            info = json.loads(info)
            tst_data.append(info['title'])
            tst_labels.append(np.array(info['target_ind']))

    assert len(trn_data) == len(trn_labels)

    trn_sents, tst_sents = data_cleaner(trn_data), data_cleaner(tst_data)

    inv_prop = get_inv_prop(trn_labels, args)

    trn_sents = pad_sentences(trn_sents, max_length=args.sequence_length)
    tst_sents = pad_sentences(tst_sents, max_length=args.sequence_length)
    trn_labels = np.array(trn_labels)
    tst_labels = np.array(tst_labels)

    vocabulary, vocabulary_inv = build_vocab(trn_sents + tst_sents, vocab_size=VOCAB_SIZE[args.dataset])

    assert vocabulary_inv[0] == '<PAD/>', "Padding word is not the first index of embeddings"

    X_trn = build_input_data(trn_sents, vocabulary)
    X_tst = build_input_data(tst_sents, vocabulary)

    del trn_data, tst_data, trn_sents, tst_sents

    embedding_weights = load_word2vec(vocabulary_inv)
    embedding_weights[0] = np.zeros(300)

    np.save(os.path.join(args.data_path, 'emb_weights'), embedding_weights)

    return X_trn, trn_labels, X_tst, tst_labels, inv_prop, embedding_weights
