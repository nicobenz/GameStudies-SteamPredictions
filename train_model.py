import json
import pickle

import h5py
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def naive_bayes(data):
    # create lists with token strings and labels
    token_list = []
    label_list = []
    for label, tokens in data.items():
        for tok in tokens:
            toks = ' '.join(tok)
            token_list.append(toks)
            label_list.append(label)

    # separate sets
    X_train, X_test, y_train, y_test = train_test_split(
        token_list,
        label_list,
        test_size=0.2,
        random_state=42
    )

    # get tf-idf from token strings
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # fit naive bayes
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)

    # predict
    y_pred = nb_classifier.predict(X_test_tfidf)

    # evaluate and save results
    evaluate_model("naive_bayes.json", y_test, y_pred)


def logistic_regression(data):
    # prepare lists for training
    X = []
    y = []
    for label, reviews in data.items():
        for review in reviews:
            X.append(review)
            y.append(label)

    # convert to np arrays
    X = np.array(X)
    y = np.array(y)

    # separate sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # fit log regression model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    # predict
    y_pred = clf.predict(X_test_scaled)

    # evaluate and save
    evaluate_model("log_reg.json", y_test, y_pred)


def evaluate_model(file_name, test, pred, print_results=True):
    eval_dict = {}

    # calculate relevant metrics
    accuracy = accuracy_score(test, pred)
    precision = precision_score(test, pred, average='weighted')
    recall = recall_score(test, pred, average='weighted')
    f1 = f1_score(test, pred, average='weighted')

    # add to dict
    eval_dict["Accuracy"] = round(accuracy, 2)
    eval_dict["Precision"] = round(precision, 2)
    eval_dict["Recall"] = round(recall, 2)
    eval_dict["F1-score"] = round(f1, 2)

    # print if desired
    if print_results:
        max_key_length = max(len(key) for key in eval_dict.keys())
        for k, v in eval_dict.items():
            print(f"{k:{max_key_length}}\t{v:.2f}")
        print("---")

    # save to disk
    save_path = f"/Volumes/Data/steam/results/{file_name}"
    with open(save_path, "w") as file_out:
        json.dump(eval_dict,file_out)


# load embeddings for log regression
with h5py.File("/Volumes/Data/steam/finished_corpus/corpus.h5", "r") as file_in:
    embeddings_data = {}

    embeddings_group = file_in["embeddings"]

    for label in embeddings_group:
        embeddings_data[label] = embeddings_group[label][:]

#with open("/Volumes/Data/steam/finished_corpus/corpus.pickle", "rb") as file_in:
#    embedding_data = pickle.load(file_in)

# load token for naive bayes
with open("/Volumes/Data/steam/finished_corpus/corpus.json", "r") as file_in:
    token_data = json.load(file_in)

# train models
naive_bayes(token_data)
#logistic_regression(embeddings_data)
