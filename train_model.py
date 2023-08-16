import json
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def naive_bayes(data):
    token_list = []
    label_list = []
    for label, tokens in data.items():
        for tok in tokens:
            toks = ' '.join(tok)
            token_list.append(toks)
            label_list.append(label)

    X_train, X_test, y_train, y_test = train_test_split(
        token_list,
        label_list,
        test_size=0.2,
        random_state=42
    )

    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the tokenized reviews to obtain TF-IDF vectors
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Initialize and train Multinomial Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)

    # Predict using the trained classifier
    y_pred = nb_classifier.predict(X_test_tfidf)

    # Evaluate the classifier
    evaluate_model("naive_bayes.json", y_test, y_pred)


def logistic_regression(data):
    X = []
    y = []
    for label, reviews in data.items():
        for review in reviews:
            avg_embedding = np.mean(review, axis=0)  # Average embeddings for the review
            X.append(avg_embedding)
            y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    # Predict using the trained classifier
    y_pred = clf.predict(X_test_scaled)

    # Evaluate the classifier
    evaluate_model("log_reg.json", y_test, y_pred)


def evaluate_model(file_name, test, pred, print_results=True):
    eval_dict = {}
    accuracy = accuracy_score(test, pred)
    precision = precision_score(test, pred, average='weighted')
    recall = recall_score(test, pred, average='weighted')
    f1 = f1_score(test, pred, average='weighted')
    eval_dict["Accuracy"] = round(accuracy, 2)
    eval_dict["Precision"] = round(precision, 2)
    eval_dict["Recall"] = round(recall, 2)
    eval_dict["F1-score"] = round(f1, 2)

    if print_results:
        max_key_length = max(len(key) for key in eval_dict.keys())
        for k, v in eval_dict.items():
            print(f"{k:{max_key_length}}\t{v:.2f}")
        print("---")

    save_path = f"/Volumes/Data/steam/results/{file_name}"
    with open(save_path, "w") as file_out:
        json.dump(eval_dict,file_out)



with open("/Volumes/Data/steam/finished_corpus/corpus.pickle", "rb") as file_in:
    embedding_data = pickle.load(file_in)

with open("/Volumes/Data/steam/finished_corpus/corpus.json", "r") as file_in:
    token_data = json.load(file_in)

naive_bayes(token_data)
logistic_regression(embedding_data)
