import json
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold
import tensorflow as tf
from collections import Counter


def prepare_tfidf_old(data, file_name, folds=5):
    # create lists with token strings and labels
    token_list = []
    label_list = []
    for label, tokens in data.items():
        for tok in tokens:
            toks = ' '.join(tok)
            token_list.append(toks)
            label_list.append(label)

    # initialise vectorizer for whole dataset
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(token_list)

    # calculate most prominent tokens for each label
    label_prominent_tokens = {}
    for label, tokens in data.items():
        combined_tokens = [' '.join(tok) for tok in tokens]  # Combine token lists into single strings
        token_tfidf_scores = tfidf_vectorizer.transform(combined_tokens).toarray()[0]
        token_indices_sorted_by_tfidf = token_tfidf_scores.argsort()[::-1]  # Sort indices in descending order of TF-IDF
        top_prominent_tokens = []
        for idx in token_indices_sorted_by_tfidf[:10]:  # Top 10 most prominent tokens
            token = tfidf_vectorizer.get_feature_names_out()[idx]
            tfidf_score = token_tfidf_scores[idx]
            top_prominent_tokens.append({"token": token, "tfidf_score": tfidf_score})
        label_prominent_tokens[label] = top_prominent_tokens

    with open(f"/Volumes/Data/steam/results/{file_name}_frequency.json", 'w') as json_file:
        json.dump(label_prominent_tokens, json_file)

    # randomised k-folds
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(token_list):
        X_train, X_val = [token_list[i] for i in train_index], [token_list[i] for i in val_index]
        y_train, y_val = [label_list[i] for i in train_index], [label_list[i] for i in val_index]

        # Initialize the TF-IDF vectorizer and fit it on the entire dataset
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(token_list)

        # Transform X_train and X_val into TF-IDF vectors
        X_train_tfidf = tfidf_vectorizer.transform(X_train)
        X_val_tfidf = tfidf_vectorizer.transform(X_val)

        yield X_train_tfidf, X_val_tfidf, y_train, y_val


def prepare_tfidf(data, folds=5):
    # create lists with token strings and labels
    token_list = []
    label_list = []
    for label, tokens in data.items():
        for tok in tokens:
            toks = ' '.join(tok)
            token_list.append(toks)
            label_list.append(label)

    # randomised k-folds
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(token_list):
        X_train, X_val = [token_list[i] for i in train_index], [token_list[i] for i in val_index]
        y_train, y_val = [label_list[i] for i in train_index], [label_list[i] for i in val_index]

        # Initialize the TF-IDF vectorizer and fit it on the entire dataset
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(token_list)

        # Transform X_train and X_val into TF-IDF vectors
        X_train_tfidf = tfidf_vectorizer.transform(X_train)
        X_val_tfidf = tfidf_vectorizer.transform(X_val)

        yield X_train_tfidf, X_val_tfidf, y_train, y_val


def calculate_prominent_tokens(data, num_of_tokens=50):
    token_list = []
    label_list = []
    for label, tokens in data.items():
        for tok in tokens:
            toks = ' '.join(tok)
            token_list.append(toks)
            label_list.append(label)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(token_list)

    # calculate most prominent tokens for each label using streaming-like approach
    label_prominent_tokens = {}

    for label, combined_tokens in data_generator(data):
        document = ' '.join(combined_tokens)  # Combine tokens into a single document
        token_tfidf_scores = tfidf_vectorizer.transform([document]).toarray()[0]
        token_indices_sorted_by_tfidf = token_tfidf_scores.argsort()[::-1]
        top_prominent_tokens = []
        for idx in token_indices_sorted_by_tfidf[:num_of_tokens]:
            token = tfidf_vectorizer.get_feature_names_out()[idx]
            tfidf_score = token_tfidf_scores[idx]
            top_prominent_tokens.append((token, round(tfidf_score, 2)))
        label_prominent_tokens[label] = top_prominent_tokens

    with open(f"/Volumes/Data/steam/results/tf-idf_frequency.json", 'w') as json_file:
        json.dump(label_prominent_tokens, json_file)


def evaluate_most_prominent_tokens_for_stopword_removal():
    with open(f"/Volumes/Data/steam/results/tf-idf_frequency.json", 'r') as json_file:
        tfidf_data = json.loads(json_file.read())

    # gather all tokens throughout all genres
    all_tokens = []
    for label, token_list in tfidf_data.items():
        for item in token_list:
            all_tokens.append(item[0])

    # count all duplicates and sort descending
    token_counts = Counter(all_tokens)
    token_counts = sorted(token_counts.items(), key=lambda item: item[1], reverse=True)

    # show only tokens that are very prominent for more than one genre
    for t, c in token_counts:
        if c > 2:
            print(t, c)


def data_generator(data):
    for label, tokens in data.items():
        yield label, [' '.join(tok) for tok in tokens]


def naive_bayes(data, folds=5):
    model_name = "naive_bayes"
    print("Naive Bayes")
    collected_metrics = []
    # iterate over folds
    for fold, (X_train, X_test, y_train, y_test) in enumerate(prepare_tfidf(data, model_name, folds), start=1):
        nb_classifier = MultinomialNB()
        print(f"Fitting fold {fold}...")
        nb_classifier.fit(X_train, y_train)
        y_pred = nb_classifier.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        collected_metrics.append(report)

    mean_folding_report(collected_metrics, model_name)


def logistic_regression(data, folds=5):
    model_name = "log_reg"
    print("Logistic Regression")
    collected_metrics = []
    # iterate over folds
    for fold, (X_train, X_test, y_train, y_test) in enumerate(prepare_tfidf(data, model_name, folds), start=1):
        nb_classifier = MultinomialNB()
        print(f"Fitting fold {fold}...")
        nb_classifier.fit(X_train, y_train)
        y_pred = nb_classifier.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        collected_metrics.append(report)

    mean_folding_report(collected_metrics, model_name)


def random_forest(data):
    model_name = "random_forest"
    print("Random Forest")
    # prepare lists for training
    X_train, X_test, y_train, y_test = prepare_tfidf(data)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Fitting...")
    clf.fit(X_train, y_train)

    print("Predicting...")
    y_pred = clf.predict(X_test)

    print("Results:")
    model_evaluation_overview(model_name, y_test, y_pred)

    report = classification_report(y_test, y_pred)

    # save classification report to a file
    with open("/Volumes/Data/steam/results/random_forest_full_report.txt", "w") as f:
        f.write(report)


def support_vector_machine(data):
    print("Support Vector Machine")
    # prepare lists for training
    X_train, X_test, y_train, y_test = prepare_tfidf(data)

    clf = SVC(kernel='linear')  # maybe try other kernels?
    print("Fitting...")
    # fit SVM model
    clf.fit(X_train, y_train)

    print("Predicting...")
    y_pred = clf.predict(X_test)

    print("Results:")
    model_evaluation_overview("svm.json", y_test, y_pred)

    report = classification_report(y_test, y_pred)

    # save classification report to a file
    with open("/Volumes/Data/steam/results/svm_full_report.txt", "w") as f:
        f.write(report)


def recurrent_neural_network(data):
    # parse data in batches for memory efficiency
    def batch_generator(features, labels, batch_size):
        num_samples = features.shape[0]
        while True:
            indices = np.random.permutation(num_samples)
            for start_idx in range(0, num_samples, batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                X_batch = tf.convert_to_tensor(features[batch_indices].toarray(), dtype=tf.float32)
                y_batch = np.array(labels)[batch_indices]
                yield X_batch, y_batch

    print("Recurrent Neural Network")
    # prepare lists for training
    X_train, X_test, y_train, y_test = prepare_tfidf(data)

    # binarize
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)

    # define model
    num_classes = y_train.shape[1]
    model = Sequential()
    model.add(Embedding(input_dim=X_train.shape[1], output_dim=64))
    model.add(LSTM(128))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Fitting...")
    batch_size = 32  # adjust maybe?
    train_generator = batch_generator(X_train, y_train, batch_size)
    steps_per_epoch = X_train.shape[0] // batch_size
    model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=10, verbose=1)

    print("Predicting...")
    y_pred_prob = model.predict(X_test)

    # get prediction lists
    y_pred = mlb.inverse_transform(y_pred_prob > 0.5)

    # evaluate and save
    print("Results:")
    model_evaluation_overview("rnn.json", y_test, y_pred)

    report = classification_report(y_test, y_pred)

    # save classification report to a file
    with open("/Volumes/Data/steam/results/rnn_full_report.txt", "w") as f:
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
        json.dump(eval_dict, file_out)


def mean_folding_report(metrics_data, filename, print_results=True):
    metrics = {}
    for item in metrics_data:
        for key, value in item.items():
            # print(key, value)
            if key == "accuracy":
                if key in metrics:
                    metrics[key].append(value)
                else:
                    metrics[key] = [value]
            else:
                if key in metrics:
                    for k, v in value.items():
                        metrics[key][k].append(v)
                else:
                    metrics[key] = {}
                    for k, v in value.items():
                        metrics[key][k] = [v]
        # print("---")
    mean_metrics = {"labels": {}, "combined": {}}
    for key, value in metrics.items():
        if key == "accuracy":
            mean_metrics["combined"][key] = round(sum(value) / len(value), 2)
        elif key == "macro avg" or key == "weighted avg":
            mean_metrics["combined"][key] = {}
            for k, v in value.items():
                mean_metrics["combined"][key][k] = round(sum(v) / len(v), 2)
        else:
            mean_metrics["labels"][key] = {}
            for k, v in value.items():
                mean_metrics["labels"][key][k] = round(sum(v) / len(v), 2)

    with open(f"/Volumes/Data/steam/results/{filename}_full_report.json", "w") as file_out:
        json.dump(mean_metrics, file_out)

    if print_results:
        print("\nMean Metrics across all folds:")
        for key, val in mean_metrics["combined"].items():
            if key == "accuracy":
                print(f"{key}: {val}")
            else:
                print(key)
                for k, v in val.items():
                    print(f"-- {k}: {v}")


# load token
with open("/Volumes/Data/steam/finished_corpus/corpus.json", "r") as file_in:
    token_data = json.load(file_in)

# calculate most prominent tokens
#calculate_prominent_tokens(token_data)
evaluate_most_prominent_tokens_for_stopword_removal()
# train models
# support_vector_machine(token_data)
# random_forest(token_data)
# logistic_regression(token_data)
# naive_bayes(token_data)
# recurrent_neural_network(token_data)

