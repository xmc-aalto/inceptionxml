from os.path import join, exists, split
import os
import numpy as np

def load_word2vec(vocabulary_inv):
    model_dir = './Glove_emb/' 

    model_name = os.path.join(model_dir, 'glove.6B.300d.txt')
    assert(os.path.exists(model_name))
    print('Loading existing Word2Vec model (Glove.6B.300d)')
    
    embedding_model = {}
    for line in open(model_name, 'r'):
        tmp = line.split(' ')
        word, vec = tmp[0], list(map(float, tmp[1:]))
        assert(len(vec) == 300)
        if word not in embedding_model:
            embedding_model[word] = vec

    embedding_weights = [embedding_model[w] if w in embedding_model
                         else np.random.uniform(-0.25, 0.25, 300)
                         for w in vocabulary_inv]
    embedding_weights = np.array(embedding_weights).astype('float32')
    
    del embedding_model
    return embedding_weights


if __name__=='__main__':
    import data_helpers
    print("Loading data...")
    x, _, _, vocabulary_inv = data_helpers.load_data()
    w = train_word2vec(x, vocabulary_inv)

