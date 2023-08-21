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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def prepare_tfidf(data, test_size=0.2):
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
        test_size=test_size,
        random_state=42
    )

    # get tf-idf from token strings
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test


def naive_bayes(data):
    print("Naive Bayes")
    X_train, X_test, y_train, y_test = prepare_tfidf(data)

    # fit naive bayes
    nb_classifier = MultinomialNB()
    print("Fitting...")
    nb_classifier.fit(X_train, y_train)

    # predict
    print("Predicting...")
    y_pred = nb_classifier.predict(X_test)

    # evaluate and save results
    print("Results:")
    model_evaluation_overview("naive_bayes.json", y_test, y_pred)

    report = classification_report(y_test, y_pred)

    # save classification report to a file
    with open("naive_bayes_full_report.txt", "w") as f:
        f.write(report)


def logistic_regression(data):
    print("Logistic Regression")
    # prepare lists for training
    X_train, X_test, y_train, y_test = prepare_tfidf(data)

    # fit log regression model
    clf = LogisticRegression(max_iter=1000)
    print("Fitting...")
    clf.fit(X_train, y_train)

    # predict
    print("Predicting...")
    y_pred = clf.predict(X_test)

    # evaluate and save
    print("Results:")
    model_evaluation_overview("log_reg.json", y_test, y_pred)

    report = classification_report(y_test, y_pred)

    # save classification report to a file
    with open("log_reg_full_report.txt", "w") as f:
        f.write(report)


def random_forest(data):
    print("Random Forest")
    # prepare lists for training
    X_train, X_test, y_train, y_test = prepare_tfidf(data)

    # fit random forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Fitting...")
    clf.fit(X_train, y_train)

    # predict
    print("Predicting...")
    y_pred = clf.predict(X_test)

    # evaluate and save
    print("Results:")
    model_evaluation_overview("random_forest.json", y_test, y_pred)

    report = classification_report(y_test, y_pred)

    # save classification report to a file
    with open("random_forest_full_report.txt", "w") as f:
        f.write(report)


def support_vector_machine(data):
    print("Support Vector Machine")
    # prepare lists for training
    X_train, X_test, y_train, y_test = prepare_tfidf(data)

    # initialize SVM with a linear kernel (you can experiment with other kernels)
    clf = SVC(kernel='linear')

    print("Fitting...")
    # fit SVM model
    clf.fit(X_train, y_train)

    print("Predicting...")
    # predict
    y_pred = clf.predict(X_test)

    print("Results:")
    # evaluate and save
    model_evaluation_overview("svm.json", y_test, y_pred)

    report = classification_report(y_test, y_pred)

    # save classification report to a file
    with open("svm_full_report.txt", "w") as f:
        f.write(report)


def model_evaluation_overview(file_name, test, pred, print_results=True):
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
support_vector_machine(token_data)
#random_forest(token_data)
#logistic_regression(token_data)
#naive_bayes(token_data)
