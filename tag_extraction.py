from bs4 import BeautifulSoup
import re
import json
import ast
from itertools import islice
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc


def exstract_tags(source_files):
    for file in tqdm(source_files):
        with open(f"{source_path}/{file}", "r") as file_in:
            app_id = file.split("_")[0]
            soup = BeautifulSoup(file_in.read(), "lxml")

            text_roi = soup.find('script', string=re.compile('.*InitAppTagModal.*'))
            text_roi = str(text_roi).replace("\n", "")
            cleaned_text = re.sub(r'\s+', '', text_roi)
            print(app_id, cleaned_text)

            cleaned_text = cleaned_text.split(f"InitAppTagModal({app_id}")[1][1:]
            cleaned_text = cleaned_text.split(",[")[0]
            tag_list = json.loads(cleaned_text)

            sorted_tags = sorted(tag_list, key=lambda x: x['count'], reverse=True)

            tag_path = "/Volumes/Data/steam/tags"
            with open(f"{tag_path}/{file}", "w") as tags_out:
                tags_out.write(str(sorted_tags))


def count_all_reviews(files):
    total_review_count = 0
    review_bars = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0, 10 100, 1000, 10000, 100000, 500000, 1000000, 1000000+
    for file in tqdm(files):
        with open(f"{review_path}/{file}") as rev_in:
            review = json.loads(rev_in.read())
            revs = review["reviews"]
            rev_count = len(revs)
            total_review_count += rev_count

            if rev_count >= 1000000:
                review_bars[8] += 1
            elif rev_count >= 500000:
                review_bars[7] += 1
            elif rev_count >= 100000:
                review_bars[6] += 1
            elif rev_count >= 10000:
                review_bars[5] += 1
            elif rev_count >= 1000:
                review_bars[4] += 1
            elif rev_count >= 100:
                review_bars[3] += 1
            elif rev_count >= 10:
                review_bars[2] += 1
            elif rev_count >= 1:
                review_bars[1] += 1
            elif rev_count == 0:
                review_bars[0] += 1
            else:
                print("something wrong")
    with open("/Volumes/Data/steam/stats/review_bars.txt", "w") as rev_out:
        for bar in review_bars:
            rev_out.write(f"{bar}\n")

    y_perc = []
    for bar in review_bars:
        if bar > 0:
            percentage = f"{round((bar / sum(review_bars)) * 100, 1)}%"
            y_perc.append(percentage)
        else:
            y_perc.append("0%")
    less = r"$<$"
    greater = r"$>$"
    x_labels = ["0", r"$<10$", f"{less}100", f"{less}1,000", f"{less}10,000", f"{less}100,000",
                f"{less}500,000", f"{less}1,000,000", f"{greater}1,000,000"]

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4))

    ax = sns.barplot(x=x_labels, y=review_bars, palette="pastel")

    for i, (v, p) in enumerate(zip(review_bars, y_perc)):
        if v > 1500:
            ax.text(i, v + (0.002*sum(review_bars)), str(v), color='black', ha='center')
            ax.text(i, v - (0.01*sum(review_bars)), p, color='white', ha='center')
        elif v > 0:
            ax.text(i, v + (0.002 * sum(review_bars)), str(v), color='black', ha='center')
        else:
            ax.text(i, v + 0.05, "", color='black', ha='center')

    plt.xlabel("Number of reviews")
    plt.ylabel("Number of games")
    plt.savefig("/Volumes/Data/steam/stats/review_plot.png", dpi=600)


def review_plot():
    with open("/Volumes/Data/steam/stats/review_bars.txt", "r") as rev_out:
        review_bars = rev_out.read()
    review_bars = review_bars.split("\n")
    review_bars = [int(r) for r in review_bars if r.isdigit()]

    y_perc = []
    for bar in review_bars:
        if bar > 0:
            percentage = f"{round((bar / sum(review_bars)) * 100, 1)}\%"  # escape char because of latex rendering
            y_perc.append(percentage)
        else:
            y_perc.append("0%")

    x_labels = ["0", "$<10$", "$<100$", "$<1,000$", "$<10,000$", "$<100,000$",
                "$<500,000$", "$<1,000,000$", "$>1,000,000$"]

    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 4))

    ax = sns.barplot(x=x_labels, y=review_bars, palette="muted")

    for i, (v, p) in enumerate(zip(review_bars, y_perc)):
        if v > 1500:
            ax.text(i, v + (0.002*sum(review_bars)), str(v), color='black', ha='center')
            ax.text(i, v - (0.02*sum(review_bars)), p, color='white', ha='center')
        elif v > 0:
            ax.text(i, v + (0.002 * sum(review_bars)), str(v), color='black', ha='center')
        else:
            ax.text(i, v + 0.05, "", color='black', ha='center')

    plt.xticks(rotation=-30, ha="left")
    plt.xlabel("Anzahl Reviews")
    plt.ylabel("Anzahl Spiele")
    plt.tight_layout()
    plt.savefig("/Volumes/Data/steam/stats/review_plot.png", dpi=600)


def count_tags(files):
    tag_dict = {}
    for file in tqdm(files):
        with open(f"/Volumes/Data/steam/tags/{file}", "r") as tag_in:
            content = tag_in.read()
            tags = ast.literal_eval(content)
            tags = sorted(tags, key=lambda x: x['count'], reverse=True)

            for tag in tags[:3]:
                if tag["name"] in tag_dict:
                    tag_dict[tag["name"]] += 1
                else:
                    tag_dict[tag["name"]] = 1

    sorted_dict = dict(sorted(tag_dict.items(), key=lambda x: x[1], reverse=True))
    for key, value in islice(sorted_dict.items(), 50):
        print(f"{value}: {key}")
    print("---")
    print(len(tag_dict))

    with open("/Volumes/Data/steam/stats/all_tags.txt", "w") as tags_out:
        tags_out.write(str(sorted_dict))


def filter_reviews(files, min_token, max_token):
    filtered_review_number = 0
    for file in tqdm(files):
        with open(f"{review_path}/{file}") as rev_in:
            review = json.loads(rev_in.read())
            revs = review["reviews"]

            if revs:
                for rev in revs:
                    if rev["review"]:
                        rev_length = len(rev["review"])
                        if rev["language"] == "english":
                            if min_token <= rev_length <= max_token:
                                filtered_review_number += 1
    with open(f"/Volumes/Data/steam/stats/filtered_reviews_{min_token}_{max_token}.txt", "w") as f:
        f.write(str(filtered_review_number))


def plot_tag_distribution(maximum):
    with open("/Volumes/Data/steam/stats/all_tags.txt", "r") as f:
        tags = f.read()
        tags = ast.literal_eval(tags)
        x = list(tags.keys())
        y = list(tags.values())
        y_p = []
        for bar in y:
            if bar > 0:
                percentage = round((bar / sum(y)) * 100, 1)
                y_p.append(percentage)
            else:
                y_p.append(0)
        y = y[:maximum]
        x = x[:maximum]
        y_p = y_p[:maximum]
        y_perc = [f"{p}\%" for p in y_p]

        sns.set(style="whitegrid")
        plt.figure(figsize=(7, 4))

        ax = sns.barplot(x=x, y=y, palette="muted")

        for i, (v, p) in enumerate(zip(y, y_perc)):
            ax.text(i, v + (0.002 * sum(y)), str(v), color='black', ha='center')
            ax.text(i, v - (0.01 * sum(y)), p, color='white', ha='center')

        plt.xticks(rotation=-30, ha="left")
        plt.ylabel("Anzahl Spiele")
        plt.tight_layout()
        plt.savefig("/Volumes/Data/steam/stats/tags_plot.png", dpi=600)


source_path = "/Volumes/Data/steam/source"

review_path = "/Volumes/Data/steam/reviews"
review_files = [f for f in listdir(review_path) if isfile(join(review_path, f)) and f != ".DS_Store"]

# Enable LaTeX rendering
rc('text', usetex=True)

# Specify the fonts
rc('font', family='serif')
rc('font', serif='Computer Modern')

plot_tag_distribution(10)
