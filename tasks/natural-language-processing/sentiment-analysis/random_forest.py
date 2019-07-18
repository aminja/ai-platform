import sys
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def vectorizer(args):
    with mlflow.start_run():
        data = pd.read_pickle(args['data_path'])
        encoded_label = pd.get_dummies(data[args['label_col']])
        label_columns = encoded_label.columns.values

        train, test = train_test_split(data, test_size=0.2, random_state=0)

        train_X = np.array(train[args['feature_column']].values.tolist())
        test_X = np.array(test[args['feature_column']].values.tolist())
        # print(train_X)
        train_y = train[label_columns].values
        test_y = test[label_columns].values

        clf = RandomForestClassifier(n_estimators=int(args['n_estimators']),
                                     max_depth=int(args['max_depth']),
                                     random_state=0)
        clf.fit(train_X, train_y)
        pred_y = clf.predict(test_X)
        print("Accuracy is: {0:.2f}".format(accuracy_score(pred_y, test_y)))



if __name__ == '__main__':
    keys = 'data_path', 'feature_column', 'label_col', 'n_estimators', 'max_depth'
    args = {k: v for k, v in zip(keys, sys.argv[1:])}
    vectorizer(args)
