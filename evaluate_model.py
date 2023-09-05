import os
import json
"""
to-do:
- model metrics table
"""

# load all model metrics
path = "data/results/model_metrics"
saved_models = [f"{path}/{x}" for x in os.listdir(path) if ".DS_Store" not in x]

model_names = []
model_metrics = []
for m in saved_models:
    name = m.split("/")[-1]
    name = name.replace("_full_report.json", "")
    name = name.split("_")
    name = [token.capitalize() for token in name]
    name = ' '.join(name)
    model_names.append(name)
    with open(m, "r") as file_in:
        content = json.load(file_in)
        aggregated = list(content["combined"]["macro avg"].values())
        model_metrics.append(aggregated)
        for keys, values in content["labels"].items():
            vals = list(values.values())
            model_metrics.append(vals)

with open("tex_templates/template_model_metrics_table.tex", "rb") as template_in:
    template = template_in.read().decode('utf-8', 'ignore')
for name in model_names:
    template = template.replace("<name>", name)
    for i, model in enumerate(model_metrics):
        for j, value in enumerate(model):
            template = template.replace(f"<{i}{j}>", str(value))

with open("tex/model_metrics_table.tex", "w", encoding="utf-8") as latex_out:
    latex_out.write(template)
