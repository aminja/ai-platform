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

import six
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in six.iteritems(parameters):
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.status != RunStatus.FINISHED:
            eprint(("Run matched, but is not FINISHED, so skipping "
                    "(run_id=%s, status=%s)") % (run_info.run_id, run_info.status))
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(("Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)") % previous_version, git_commit)
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
        return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


def workflow(args):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        data_handler_run = _get_or_run("data_handler", args, git_commit)
        input_data_uri = os.path.join(data_handler_run.info.artifact_uri, 'processed_data_dir', 'input_data.pkl')
        word2vec_vectorizer_run = _get_or_run('word2vec_vectorizer',
                                              {'data_path': input_data_uri,
                                               'text_col' : args['text_col'],
                                               'dimension': 30},
                                              git_commit)
        vect_data_uri = os.path.join(word2vec_vectorizer_run.info.artifact_uri,
                                     'processed_data_dir', 'vectorized_data.pkl')
        rf_params = {
            'data_path': vect_data_uri,
            'label_col': 'airline_sentiment',
            'n_estimators': 500,
            'max_depth': 10
        }
        random_forest_run = _get_or_run("random_forest",
                                        rf_params,
                                        git_commit)


if __name__ == '__main__':
    keys = 'data_path', 'text_col', 'label_col', 'punctuation'
    args = {k: v for k, v in zip(keys, sys.argv[1:])}
    workflow(args)
