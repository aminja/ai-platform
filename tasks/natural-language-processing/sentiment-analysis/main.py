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


def workflow(m):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tweets.csv")
        data = pd.read_csv(file_path)
        print(data.head(m))


if __name__ == '__main__':
    m = sys.argv[1]
    workflow(int(m))