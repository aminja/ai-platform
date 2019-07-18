import nltk
import os
import sys
import mlflow
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('punkt')
nltk.download('stopwords')


def data_loader(file_path, text_col, label_col):
    '''
    Load the desired columns from input csv file
    :param file_path: path of the input file
    :param text_col: the input column for NLP
    :param label_col: the column containing true labels (sentiments)
    :return: pandas dataframe
    '''
    try:
        raw_data = pd.read_csv(file_path)
    except FileNotFoundError as fnf_error:
        raise fnf_error

    data = raw_data[[text_col, label_col]]
    encoded_label = pd.get_dummies(data[label_col])
    data = pd.concat((data, encoded_label), 1)
    data = text_processing(data, text_col,
                           punctuation=True if args['punctuation'] == '1' else False)
    return data


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


def data_handler(args):
    with mlflow.start_run():
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args['data_path'])
        print(file_path)
        data = data_loader(file_path, args['text_col'], args['label_col'])

        data.to_pickle('input_data.pkl')
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_data.pkl')

        print("Uploading processed data")
        mlflow.log_artifact(data_path, 'processed_data_dir')


if __name__ == '__main__':
    keys = 'data_path', 'text_col', 'label_col', 'punctuation'
    args = {k: v for k, v in zip(keys, sys.argv[1:])}
    data_handler(args)
