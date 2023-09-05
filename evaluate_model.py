import os
"""
to-do:
- model metrics table
"""

# load all model metrics
path = "data/results/model_metrics"
saved_models = [f"{path}/{x}" for x in os.listdir(path) if ".DS_Store" not in x]

aggregation_metrics = []
model_names = []
model_metrics = []
for m in saved_models:
    name = m.split("/")[-1]
    name = name.replace("_full_report.txt", "")
    name = name.split("_")
    name = [token.capitalize() for token in name]
    name = ' '.join(name)
    model_names.append(name)

    with open(m, "r") as file_in:
        content = file_in.read()
        content = content.split("\n")
        for idx, cont in enumerate(content):
            if idx != 0:
                split_cont = cont.split(",")
                split_cont.pop(0)
                if split_cont:
                    model_metrics.append(split_cont)

with open("tex_templates/template_model_metrics_table.tex", "rb") as template_in:
    template = template_in.read().decode('utf-8', 'ignore')
for name in model_names:
    template = template.replace("<name>", name)
    for i, model in enumerate(model_metrics):
        for j, value in enumerate(model):
            template = template.replace(f"<{i}{j}>", value)

with open("tex/model_metrics_table.tex", "w", encoding="utf-8") as latex_out:
    latex_out.write(template)
