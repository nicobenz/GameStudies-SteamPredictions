import os
import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns


"""
to-do:
- model metrics table
"""


def roc(source_path):
    # set seaborn stuff
    sns.set_theme()
    sns.set_context("paper")
    custom_palette = sns.color_palette("Spectral")
    sns.set_palette(custom_palette)

    # prepare file loading
    file_path = f"{source_path}/results/model_predictions"
    pred_files = [file for file in os.listdir(file_path) if ".DS_Store" not in file]
    pred_files = [f"{file_path}/{file}" for file in pred_files]

    # loop over all prediction files
    for file in pred_files:
        print(file)
        with open(file, "r") as pred_in:
            predictions = json.load(pred_in)
        save_name = file.split("/")[-1].replace("_predictions.json", "")
        print(save_name)
        model_name = save_name.split("_")
        model_name = [x.capitalize() for x in model_name]
        model_name = " ".join(model_name)

        # prepare data for roc
        all_labels = set(label for fold in predictions["actual"] for label in fold)
        classes = sorted(list(all_labels))  # Sort for consistency

        roc_curves_per_class = []
        roc_auc_per_class = []

        # loop for all labels
        for class_name in classes:
            roc_curves_class = []
            roc_auc_class = []

            # loop for each fold
            for fold_index in range(len(predictions["actual"])):
                test = predictions["actual"][fold_index]
                pred_probs = predictions["probability"][fold_index]

                # get probabilities
                class_index = classes.index(class_name)
                actual_probs = [prob[class_index] for prob in pred_probs]
                actual_labels = [1 if class_name in test_labels else 0 for test_labels in test]

                # get fale positive rate and true positive rate
                fpr, tpr, _ = roc_curve(actual_labels, actual_probs)
                roc_auc = auc(fpr, tpr)

                roc_curves_class.append((fpr, tpr))
                roc_auc_class.append(roc_auc)
            roc_curves_per_class.append(roc_curves_class)
            roc_auc_per_class.append(roc_auc_class)

        # plot
        plt.figure(figsize=(8, 6))
        for class_index, roc_curves_class in enumerate(roc_curves_per_class):
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.zeros_like(mean_fpr)

            for fpr, tpr in roc_curves_class:
                mean_tpr += np.interp(mean_fpr, fpr, tpr)

            mean_tpr /= len(roc_curves_class)

            label = f'{classes[class_index]} (AUC = {np.mean(roc_auc_per_class[class_index]):.2f})'
            sns.lineplot(x=mean_fpr, y=mean_tpr, lw=2, label=label)

        plt.axline([0, 0], [1, 1], color="gray", lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic One-vs-Rest (ROC OvR) for {model_name}')
        plt.legend(loc='lower right')
        plt.savefig(f"{source_path}/results/plots/{save_name}_roc.png", dpi=600)


def put_metrics_into_latex_tables():
    # table for all models and all labels
    path = "data/results/model_metrics"
    saved_models = [f"{path}/{x}" for x in os.listdir(path) if ".DS_Store" not in x]
    saved_models.sort()

    model_names = []
    model_metrics = []
    for idx, m in enumerate(saved_models, start=1):
        model_data = []
        name = m.split("/")[-1]
        name = name.replace("_full_report.json", "")
        name = name.replace(f"{idx}_", "")
        name = name.split("_")
        name = [token.capitalize() for token in name]
        name = ' '.join(name)
        model_names.append(name)
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
        template = template.replace("<name>", name)
        for i, mod in enumerate(model):
            for j, value in enumerate(mod):
                template = template.replace(f"<{i}{j}>", str(value))
        file_label = name.split(" ")
        file_label = [label.lower() for label in file_label]
        file_label = "_".join(file_label)

        with open(f"tex/model_metrics_table_{file_label}.tex", "w", encoding="utf-8") as latex_out:
            latex_out.write(template)

    # table for mean values across models
    with open("tex_templates/template_model_aggregation_metrics_table.tex", "rb") as template_in:
        template = template_in.read().decode('utf-8', 'ignore')

    model_mean = [model[0] for model in model_metrics]
    num_sublists = len(model_mean)
    num_elements = len(model_mean[0])
    means = [sum(model_mean[i][j] for i in range(num_sublists)) / num_sublists for j in range(num_elements)]
    means = [round(mean, 2) for mean in means]
    model_mean.append(means)

    for i, model in enumerate(model_mean):
        for j, vals in enumerate(model):
            template = template.replace(f"<{i}{j}>", str(vals))
    with open(f"tex/model_aggregation_metrics_table.tex", "w", encoding="utf-8") as latex_out:
        latex_out.write(template)


#put_metrics_into_latex_tables()
roc("data")
