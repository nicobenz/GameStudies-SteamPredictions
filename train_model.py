import json
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
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
from imblearn.over_sampling import SMOTE  # currently disabled


def calculate_prominent_tokens(data, source_path, num_of_tokens=50):
    token_list = []
    label_list = []
    for label, tokens in data.items():
        for tok in tokens:
            toks = ' '.join(tok)
            token_list.append(toks)
            label_list.append(label)

    tfidf_vectorizer = TfidfVectorizer(stop_words=None)
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

    with open(f"{source_path}/results/tf-idf_frequency.json", 'w') as json_file:
        json.dump(label_prominent_tokens, json_file)


def evaluate_most_prominent_tokens_for_stopword_removal(source_path, print_result=True):
    with open(f"{source_path}/results/tf-idf_frequency.json", 'r') as json_file:
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
    most_prominent_tokens = {}
    for t, c in token_counts:
        if c > 2:
            most_prominent_tokens[t] = c
            if print_result:
                print(t, c)

    with open(f"{source_path}/results/most_common_tokens.json", 'w') as json_file:
        json.dump(most_prominent_tokens, json_file)


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

        # initialize vectorizer and fit fold
        tfidf_vectorizer = TfidfVectorizer(stop_words=None)
        tfidf_vectorizer.fit(X_train)

        # transform to tf-idf vectors
        X_train_tfidf = tfidf_vectorizer.transform(X_train)
        X_val_tfidf = tfidf_vectorizer.transform(X_val)

        # apply smote for balance (only on training set though)
        #smote = SMOTE(sampling_strategy='auto', random_state=42)
        #X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
        # smote not possible because of computational constraints

        yield X_train_tfidf, X_val_tfidf, y_train, y_val


def data_generator(data):
    for label, tokens in data.items():
        yield label, [' '.join(tok) for tok in tokens]


def train_model(data, model_name, source_path, save_order, verbose=True, learning_curve=False, folds=5):
    print(model_name)
    # create string for saving results later
    save_string = model_name.split(" ")
    save_string = [x.lower() for x in save_string]
    save_string = '_'.join(save_string)

    # lists for learning curve data
    learning_curve_train_sizes = []
    learning_curve_train_scores = []
    learning_curve_test_scores = []

    collected_metrics = []
    collected_predictions = {"predicted": [], "probability": [], "actual": []}
    confusion_matrices = []
    # iterate over folds
    for fold, (X_train, X_test, y_train, y_test) in enumerate(prepare_tfidf(data, folds), start=1):
        if save_string == "naive_bayes":
            classifier = MultinomialNB(
                alpha=0.5
            )
        elif save_string == "logistic_regression":
            classifier = LogisticRegression(
                max_iter=1000,
                C=0.5,
                verbose=verbose
            )
        elif save_string == "random_forest":
            classifier = RandomForestClassifier(
                random_state=42,
                verbose=verbose
            )
        elif save_string == "support_vector_machine":
            classifier = SVC(kernel='linear', probability=True, verbose=verbose)
        else:
            classifier = MultinomialNB()
        print(f"Fitting fold {fold}...")
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_probs = classifier.predict_proba(X_test)

        if learning_curve:
            train_sizes, train_scores, test_scores = generate_learning_curve(classifier, X_train, y_train, X_test, y_test)
            learning_curve_train_sizes.append(train_sizes)
            learning_curve_train_scores.append(train_scores)
            learning_curve_test_scores.append(test_scores)

        report = classification_report(y_test, y_pred, output_dict=True)
        conf_mat = confusion_matrix(y_test, y_pred, normalize='true')

        confusion_matrices.append(conf_mat)
        collected_metrics.append(report)
        collected_predictions["predicted"].append(y_pred)
        collected_predictions["probability"].append(y_probs)
        collected_predictions["actual"].append(y_test)

    mean_conf_mat = np.mean(confusion_matrices, axis=0)
    np.savetxt(f"{source_path}/results/confusion_matrices/{save_order}_{save_string}_conf_mat.txt", mean_conf_mat, fmt='%.2f', delimiter='\t')
    mean_folding_report(collected_metrics, save_string, source_path, save_order)

    convert = list(collected_predictions.keys())

    for key in convert:
        for i, item in enumerate(collected_predictions[key]):
            if isinstance(item, np.ndarray):
                collected_predictions[key][i] = item.tolist()

    with open(f"{source_path}/results/model_predictions/{save_string}_predictions.json", "w", encoding="utf-8") as pred_out:
        json.dump(collected_predictions, pred_out)

    with open(f"{source_path}/results/model_metrics/folds/{save_order}_{save_string}_fold_report.json", "w", encoding="utf-8") as pred_out:
        json.dump(collected_metrics, pred_out)

    if learning_curve:
        learning_curve_data = {
            'train_sizes': learning_curve_train_sizes,
            'train_scores': learning_curve_train_scores,
            'test_scores': learning_curve_test_scores,
        }
        with open(f"{source_path}/results/learning_curves/{save_string}_learning_curve.json", "w",
                  encoding="utf-8") as lc_out:
            json.dump(learning_curve_data, lc_out)


def generate_learning_curve(classifier, X_train, y_train, X_test, y_test, step_size=100):
    # currently not used because of large increase in computational demands
    train_sizes = []
    train_scores = []
    test_scores = []

    num_samples, _ = X_train.shape

    for step in range(step_size, num_samples, step_size):
        X_subset = X_train[:step]
        y_subset = y_train[:step]

        classifier.fit(X_subset, y_subset)

        train_pred = classifier.predict(X_subset)
        test_pred = classifier.predict(X_test)

        train_accuracy = accuracy_score(y_subset, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        train_sizes.append(step)
        train_scores.append(train_accuracy)
        test_scores.append(test_accuracy)

    return train_sizes, train_scores, test_scores


def recurrent_neural_network(data):  # not used for final paper because of computational constraints
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


def mean_folding_report(metrics_data, filename, source_path, save_order, print_results=True):
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

    with open(f"{source_path}/results/model_metrics/full/{save_order}_{filename}_full_report.json", "w") as file_out:
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


load_locally = True
if load_locally:
    path = "data"
else:
    path = "/Volumes/Data/steam"

# load token
with open(f"{path}/finished_corpus/corpora/corpus-1-AdventureStrategySimulationRPGPuzzle_cleaned.json", "r") as file_in:
    token_data = json.load(file_in)

# calculate most prominent tokens
#calculate_prominent_tokens(token_data, path)
#evaluate_most_prominent_tokens_for_stopword_removal(path)

# train models
train_model(token_data, "Naive Bayes", path, 1, verbose=False)
train_model(token_data, "Logistic Regression", path, 2, verbose=False)
train_model(token_data, "Random Forest", path, 3, verbose=False)
#train_model(token_data, "Support Vector Machine", path, 4)  # not used because of computational demands
