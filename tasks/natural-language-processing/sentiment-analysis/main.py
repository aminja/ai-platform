import nltk
import os
import sys
import mlflow
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')


def text_processing(df, col_name, punctuation=True):
    '''
    Given dataframe and desired column to convert the raw text into processed tokens
    :param df: dataframe
    :param col_name: column name
    :param punctuation: the boolean which shows whether punctuation cleaning applies or not
    :return: processed dataframe
    '''
    if not punctuation:
        # punctuation cleaning
        df['clean_{}'.format(col_name)] = df[col_name].str.replace('[^\w\s]','')

        # tokenizing phase
        df['tokens_{}'.format(col_name)] = df['clean_{}'.format(col_name)].apply(nltk.word_tokenize)
    else:
        # tokenizing phase
        df['tokens_{}'.format(col_name)] = df[col_name].apply(nltk.word_tokenize)

    # stopwords cleaning
    stop = stopwords.words('english')
    df['stop_{}'.format(col_name)] = df['tokens_{}'.format(col_name)]. \
        apply(lambda x: [word for word in x if word not in (stop)])

    # stemming
    ps = PorterStemmer()
    df['stemmed_{}'.format(col_name)] = df['stop_{}'.format(col_name)]. \
        apply(lambda words: [ps.stem(word) for word in words])

    return df


def workflow(m):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tweets.csv")
        raw_data = pd.read_csv(file_path)
        data = raw_data[['tweet_id', 'airline_sentiment', 'text']]
        data = pd.concat((data, pd.get_dummies(data['airline_sentiment'])), 1)
        data = text_processing(data, 'text', punctuation=True)
        # vectorization
        dim = 30
        model = Word2Vec(data['stemmed_text'], size=dim,
                         window=5, min_count=3,
                         workers=1, sample=1e-3, seed=0)
        vectors = model.wv
        train, test = train_test_split(data, test_size=0.2, random_state=0)

        def UDF(wlist):
            out = np.array([0.0] * dim)
            for word in wlist:
                if word in vectors.vocab:
                    out += np.array(vectors[word])
                else:
                    pass
            return out  # /len(wlist)

        # Word2vec
        train_X = train['stemmed_text'].apply(lambda wlist: UDF(wlist))
        test_X = test['stemmed_text'].apply(lambda wlist: UDF(wlist))

        train_X = np.array(train_X.values.tolist())
        test_X = np.array(test_X.values.tolist())
        train_y = train[['negative', 'neutral', 'positive']].values
        test_y = test[['negative', 'neutral', 'positive']].values

        clf = RandomForestClassifier(n_estimators=500, max_depth=40, random_state=0)
        clf.fit(train_X, train_y)
        pred_y = clf.predict(test_X)
        print("Accuracy is: {0:.2f}%".format(accuracy_score(pred_y, test_y)))


if __name__ == '__main__':
    m = sys.argv[1]
    workflow(int(m))