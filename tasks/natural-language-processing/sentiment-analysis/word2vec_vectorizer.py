import os
import sys
import mlflow
import pandas as pd
import numpy as np
from gensim.models import Word2Vec


def sentence_vectorizer(vectors, wlist, dim):
    # Aggregate vectors of words
    # e.g sentence2vector
    out = np.array([0.0] * dim)
    for word in wlist:
        if word in vectors.vocab:
            out += np.array(vectors[word])
    else:
        pass
    return out


def vectorizer(args):
    with mlflow.start_run():
        dim = int(args['dimension'])
        data = pd.read_pickle(args['data_path'])

        model = Word2Vec(data['stemmed_{}'.format(args['text_col'])], size=dim,
                         window=5, min_count=3,
                         workers=1, sample=1e-3, seed=0)
        vectors = model.wv
        data['vectorized'] = data['stemmed_{}'.format(args['text_col'])].apply(lambda wlist: sentence_vectorizer(vectors, wlist, dim))
        data.to_pickle('vectorized_data.pkl')
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vectorized_data.pkl')

        print("Uploading vectorized data")
        mlflow.log_artifact(data_path, 'processed_data_dir')


if __name__ == '__main__':
    keys = 'data_path', 'text_col', 'dimension'
    args = {k: v for k, v in zip(keys, sys.argv[1:])}
    vectorizer(args)
