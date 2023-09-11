import os
import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns
import locale


def save_tex(file_name, template):
    with open(f"tex/{file_name}", "w", encoding="utf-8") as latex_out:
        latex_out.write(template)


def load_tex(file_name):
    with open(f"tex_templates/{file_name}", "rb") as template_in:
        template = template_in.read().decode('utf-8', 'ignore')
    return template


def format_numbers_for_readability(number_list):
    number_list = [
        locale.format_string("%.2f", number, grouping=True) if isinstance(number, float) else
        locale.format_string("%d", number, grouping=True) for number in number_list
    ]
    return number_list


def create_combined_table(directory, output_name):
    path = f"data/results/model_metrics"
    files = [file for file in os.listdir(f"{path}/{directory}") if ".DS_Store" not in file]
    files.sort()
    collected_values = []
    for model in files:
        collected_model = []
        with open(f"{path}/{directory}/{model}", "r") as f:
            content = json.loads(f.read())
        if directory == "full":
            collected_model.append(list(content["combined"]["macro avg"].values()))
            for key in list(content["labels"].keys()):
                collected_model.append(list(content["labels"][key].values()))
            collected_values.append(collected_model)
        else:
            model_full = model.split("_")
            model_full[3] = "full"
            model_full = "_".join(model_full)
            with open(f"{path}/full/{model_full}", "r") as f:
                content = json.loads(f.read())
            collected_model.append(list(content["combined"]["macro avg"].values()))
            with open(f"{path}/{directory}/{model}", "r") as f:
                content = json.loads(f.read())
            for genre_dict in content:
                for key in list(genre_dict.keys()):
                    if key != "accuracy" and "avg" not in key:
                        values = list(genre_dict[key].values())
                        values = [round(val, 2) for val in values]
                        collected_model.append(values)
            collected_values.append(collected_model)
    fill_stacked_latex_model_table(collected_values, output_name)


def fill_stacked_latex_model_table(stacked_list, file_name):
    """
    stacked_list = [
        [
            ["1a", 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ],
        [
            ["1b", 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ],
        [
            ["1c", 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ]
    ]
    """
    template = load_tex(f"template_{file_name}")
    for i, sub in enumerate(stacked_list):
        for j, subsub in enumerate(sub):
            for k, subsubsub in enumerate(subsub):
                template = template.replace(f"<{i}{j}{k}>", str(subsubsub))

    save_tex(file_name, template)


def stacked_roc(source_path):
    # set seaborn stuff
    sns.set_theme()
    sns.set_context("paper")
    custom_palette = sns.color_palette("Spectral")
    sns.set_palette(custom_palette)

    # prepare file loading
    file_path = f"{source_path}/results/model_predictions"
    pred_files = [file for file in os.listdir(file_path) if ".DS_Store" not in file]
    pred_files = [f"{file_path}/{file}" for file in pred_files]

    fig, axs = plt.subplots(len(pred_files), 2, figsize=(12, 6 * len(pred_files)))

    # loop over all prediction files
    for row, file in enumerate(pred_files):
        with open(file, "r") as pred_in:
            predictions = json.load(pred_in)
        save_name = file.split("/")[-1].replace("_predictions.json", "")
        model_name = save_name.split("_")
        model_name = [x.capitalize() for x in model_name]
        model_name = " ".join(model_name)

        # prepare data for roc
        all_labels = set(label for fold in predictions["actual"] for label in fold)
        classes = sorted(list(all_labels))  # Sort for consistency
        all_roc_curves = []
        all_roc_auc = []
        all_precision_values = []
        all_recall_values = []

        # loop for all labels
        for class_name in classes:
            roc_curves_per_class = []
            roc_auc_per_class = []

            # Initialize arrays to store precision and recall values
            precision_values = []
            recall_values = []

            # loop for each fold
            for fold_index in range(len(predictions["actual"])):
                test = predictions["actual"][fold_index]
                pred_probs = predictions["probability"][fold_index]
                pred_labels = predictions["predicted"][fold_index]

                # get probabilities
                class_index = classes.index(class_name)
                actual_probs = [prob[class_index] for prob in pred_probs]
                pred_labels = [1 if class_name in test_labels else 0 for test_labels in pred_labels]
                actual_labels = [1 if class_name in test_labels else 0 for test_labels in test]

                roc_curves_class = []  # Initialize roc_curves_class here
                roc_auc_class = []  # Initialize roc_auc_class here

                # ROC curve calculations
                fpr, tpr, _ = roc_curve(actual_labels, actual_probs)
                roc_auc = auc(fpr, tpr)

                roc_curves_class.append((fpr, tpr))
                roc_auc_class.append(roc_auc)

                # Calculate the area under the precision-recall curve (optional)
                precision, recall, _ = precision_recall_curve(actual_labels, actual_probs)
                precision_values.append(precision)
                recall_values.append(recall)
                # Append the roc_curves_class and roc_auc_class for this fold
                roc_curves_per_class.append(roc_curves_class)
                roc_auc_per_class.append(roc_auc_class)

                # Append the precision-recall values for this class to the list
            all_precision_values.append(precision_values)
            all_recall_values.append(recall_values)
            all_roc_curves.append(roc_curves_per_class)
            all_roc_auc.append(roc_auc_per_class)

        # plot
        for class_index, (roc_curves_class, auc_curves) in enumerate(zip(all_roc_curves, all_roc_auc)):
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.zeros_like(mean_fpr)
            for fpr, tpr in roc_curves_class[class_index]:
                mean_tpr += np.interp(mean_fpr, fpr, tpr)

            label = f"{classes[class_index]}, (AUC = {round(auc_curves[class_index][0], 2)})"
            sns.lineplot(x=mean_fpr, y=mean_tpr, lw=2, label=label, ax=axs[row, 0])

        axs[row, 0].axline([0, 0], [1, 1], color="gray", lw=2, linestyle='--')
        axs[row, 0].set_xlim([0.0, 1.0])
        axs[row, 0].set_ylim([0.0, 1.05])
        axs[row, 0].set_title(f'{model_name}\nROC Curves', fontsize=12, loc='center')
        axs[row, 0].set_xlabel('False Positive Rate')
        axs[row, 0].set_ylabel('True Positive Rate')
        axs[row, 0].legend(loc='lower right')

        for curve_index, (precision_vals, recall_vals) in enumerate(zip(all_precision_values, all_recall_values)):
            average_precision = auc(recall_vals[curve_index], precision_vals[curve_index])

            label = f'{classes[curve_index]} (AP = {average_precision:.2f})'

            sns.lineplot(x=recall_vals[curve_index], y=precision_vals[curve_index], lw=2, label=label, ax=axs[row, 1])

        axs[row, 1].set_xlim([0.0, 1.0])
        axs[row, 1].set_ylim([0.0, 1.05])
        axs[row, 1].set_title(f'{model_name}\nPrecision-Recall Curves', fontsize=12, loc='center')
        axs[row, 1].set_xlabel('Recall')
        axs[row, 1].set_ylabel('Precision')
        axs[row, 1].legend(loc='lower right')

    plt.title("ROC and Precision Recall Curves")
    plt.tight_layout()
    plt.savefig(f"{source_path}/results/plots/combined_roc.pdf")


def create_heatmap_from_confusion_matrices(source_path):
    matrix_files = [file for file in os.listdir(source_path) if ".DS_Store" not in file]
    matrix_files.sort()
    confusion_matrices = []
    for file in matrix_files:
        mat = np.loadtxt(f"{source_path}/{file}", dtype=float, delimiter='\t')
        confusion_matrices.append(mat)
    mean_matrix = np.mean(confusion_matrices, axis=0)
    mean_matrix = np.round(mean_matrix, 2)

    class_labels = ["Adventure", "Strategy", "Simulation", "RPG", "Puzzle"]

    sns.set_theme()
    sns.set_context("paper")
    custom_palette = sns.color_palette("Spectral")
    sns.set_palette(custom_palette)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust the font scale as needed
    sns.heatmap(mean_matrix, cmap="Blues", annot=True, fmt=".2f",
                xticklabels=class_labels, yticklabels=class_labels, cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f"data/results/plots/mean_confusion_matrix.pdf")



def plot_aggregated_learning_curve(source_path, save_string):
    with open(f"{source_path}/results/learning_curve_data/{save_string}_learning_curve_data.json", "r",
              encoding="utf-8") as lc_in:
        learning_curve_data = json.load(lc_in)

    # Calculate the aggregated learning curve
    aggregated_train_sizes = np.mean(learning_curve_data['train_sizes'], axis=0)
    aggregated_train_scores = np.mean(learning_curve_data['train_scores'], axis=0)
    aggregated_test_scores = np.mean(learning_curve_data['test_scores'], axis=0)

    # Plot the aggregated learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(aggregated_train_sizes, aggregated_train_scores, label='Training Accuracy', marker='o')
    plt.plot(aggregated_train_sizes, aggregated_test_scores, label='Test Accuracy', marker='o')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.title('Aggregated Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{source_path}/results/plots/{save_string}_learning_curve.pdf")


def put_fold_metrics_into_latex_tables():
    # table for all models and all labels
    path_full = "data/results/model_metrics/full"
    saved_models = [f"{path_full}/{x}" for x in os.listdir(path_full) if ".DS_Store" not in x]
    saved_models.sort()

    path_folds = "data/results/model_metrics/folds"
    saved_folds = [f"{path_folds}/{x}" for x in os.listdir(path_folds) if ".DS_Store" not in x]
    saved_folds.sort()

    for idx, (full_metrics, fold_metrics) in enumerate(zip(saved_models, saved_folds), start=1):
        name = full_metrics.split("/")[-1]
        name = name.replace(f"{idx}_", "")
        name = name.replace("_full_report.json", "")
        template = load_tex("template_combined_fold_metrics.tex")
        with open(full_metrics, "r") as file_in:
            full_content = json.loads(file_in.read())
        with open(fold_metrics, "r") as file_in:
            fold_content = json.loads(file_in.read())
        collected_lists = []
        full_items = list(full_content["combined"]["macro avg"].values())
        collected_lists.append(full_items)
        for i, fold in enumerate(fold_content):
            fold_items = list(fold["macro avg"].values())
            fold_items = [round(item, 2) for item in fold_items]
            collected_lists.append(fold_items)
        for i, items in enumerate(collected_lists):
            for j, v in enumerate(items):
                template = template.replace(f"<{i}{j}>", str(v))
        save_tex(f"fold_metrics_{name}.tex", template)


def put_model_metrics_into_latex_tables():
    # table for all models and all labels
    path = "data/results/model_metrics/full"
    saved_models = [f"{path}/{x}" for x in os.listdir(path) if ".DS_Store" not in x]
    saved_models.sort()

    model_names = []
    model_metrics = []
    for idx, m in enumerate(saved_models, start=1):
        model_data = []
        name = m.split("/")[-1]
        name = name.replace("_full_report.json", "")
        fig_name = name.split("_")[1:]
        fig_name = "_".join(fig_name)
        name = name.replace(f"{idx}_", "")
        name = name.split("_")
        name = [token.capitalize() for token in name]
        name = ' '.join(name)
        model_names.append((name, fig_name))
        with open(m, "r") as file_in:
            content = json.load(file_in)
            aggregated = list(content["combined"]["macro avg"].values())
            model_data.append(aggregated)
            for keys, values in content["labels"].items():
                vals = list(values.values())
                model_data.append(vals)
        model_metrics.append(model_data)

    for name, model in zip(model_names, model_metrics):
        with open("tex_templates/template_model_metrics_table.tex", "rb") as template_in:
            template = template_in.read().decode('utf-8', 'ignore')
        template = template.replace("tab:model_metrics", f"tab:model_metrics_{name[1]}")
        template = template.replace("<name>", name[0])
        for i, mod in enumerate(model):
            for j, value in enumerate(mod):
                template = template.replace(f"<{i}{j}>", str(value))
        file_label = name[0].split(" ")
        file_label = [label.lower() for label in file_label]
        file_label = "_".join(file_label)

        #with open(f"tex/model_metrics_table_{file_label}.tex", "w", encoding="utf-8") as latex_out:
        #    latex_out.write(template)

    # table for mean values across models
    with open("tex_templates/template_model_aggregation_metrics_table.tex", "rb") as template_in:
        template = template_in.read().decode('utf-8', 'ignore')

    model_mean = [model[0] for model in model_metrics]
    num_sublists = len(model_mean)
    num_elements = len(model_mean[0])
    means = [sum(model_mean[i][j] for i in range(num_sublists)) / num_sublists for j in range(num_elements)]
    means = [round(mean, 2) for mean in means]
    model_mean.append(means)
    # move the last element to the front
    model_mean = [model_mean[-1]] + model_mean[:-1]

    for i, model in enumerate(model_mean):
        for j, vals in enumerate(model):
            template = template.replace(f"<{i}{j}>", str(vals))
    with open(f"tex/model_aggregation_metrics_table.tex", "w", encoding="utf-8") as latex_out:
        latex_out.write(template)


def put_review_metrics_into_latex_table():
    file_names = [
        "/Users/nico/code/GameStudies-SteamPredictions/data/results/descriptive_stats/all_token_metrics.json",
        "/Users/nico/code/GameStudies-SteamPredictions/data/results/descriptive_stats/english_token_metrics.json"
    ]
    collected_metrics = []
    for file in file_names:
        with open(file, "r") as file_in:
            metrics = json.loads(file_in.read())

        metrics = list(metrics.values())
        metrics = format_numbers_for_readability(metrics)
        collected_metrics.append(metrics)

    template = load_tex("template_review_metrics.tex")

    for i, column in enumerate(collected_metrics):
        for j, metric in enumerate(column):
            template = template.replace(f"<{i}{j}>", str(metric))
    save_tex("review_metrics.tex", template)


def put_tf_idf_values_into_latex_table(length):
    with open("data/results/tf-idf_frequency.json", "r") as file_in:
        tf_idf_values = json.loads(file_in.read())
    values_list = []
    for genre in list(tf_idf_values.keys()):
        selected_tokens = tf_idf_values[genre][:length]
        values_list.append(selected_tokens)
    template = load_tex("template_tfidf_by_genre.tex")
    for i, val in enumerate(values_list):
        for j, v in enumerate(val):
            template = template.replace(f"<{i}{j}-1>", v[0])
            template = template.replace(f"<{i}{j}-2>", str(v[1]))

    save_tex("tfidf_by_genre.tex", template)


def most_prominent_token_across_genres(length):
    with open("data/results/tf-idf_frequency.json", "r") as file_in:
        tf_idf_values = json.loads(file_in.read())
    values_collection = {}
    for items in list(tf_idf_values.values()):
        for tup in items:
            if tup[0] in values_collection:
                values_collection[tup[0]] += tup[1]
            else:
                values_collection[tup[0]] = tup[1]
    values_collection = dict(sorted(values_collection.items(), key=lambda item: item[1], reverse=True))
    for key, val in values_collection.items():
        values_collection[key] = round(val, 2)
    selected_keys = []
    for i in range(length):
        selected_keys.append(list(values_collection.keys())[i])
    scores = []
    for key in selected_keys:
        score_per_genre = []
        for genre in list(tf_idf_values.keys()):
            genre_dict = {key: value for key, value in tf_idf_values[genre]}
            if key in genre_dict:
                score_per_genre.append(genre_dict[key])
            else:
                score_per_genre.append("-")
        scores.append(score_per_genre)
    table_list = []
    for key, genre in zip(selected_keys, scores):
        table_list.append([key, genre[0], genre[1], genre[2], genre[3], genre[4]])  # very ugly, change later
    table_turned = []
    for i in range(6):
        select = []
        for sublist in table_list:
            select.append(sublist[i])
        table_turned.append(select)
    template = load_tex("template_prominent_tokens.tex")
    for i, sublist in enumerate(table_turned):
        for j, val in enumerate(sublist):
            template = template.replace(f"<{i}{j}>", str(val))

    save_tex("prominent_tokens.tex", template)


locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

put_model_metrics_into_latex_tables()
stacked_roc("data")
put_review_metrics_into_latex_table()
put_tf_idf_values_into_latex_table(15)
most_prominent_token_across_genres(10)
put_fold_metrics_into_latex_tables()
create_combined_table("full", "combined_model_metrics.tex")
create_combined_table("folds", "combined_fold_metrics.tex")
create_heatmap_from_confusion_matrices("data/results/confusion_matrices")
