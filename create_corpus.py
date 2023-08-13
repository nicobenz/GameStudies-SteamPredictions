"""
create corpus
goals:
- use maximum amount of data without creating bias
- even distribution of reviews across tags
- set maximum amount of reviews possible for each game
- tags: adventure, strategy, simulation, rpg, puzzle, (sports?)
"""

from os import listdir
from os.path import isfile, join
import ast
import json
from tqdm import tqdm
import logging
import random
import spacy
import matplotlib.pyplot as plt
import squarify
import seaborn as sns


def sort_appid_by_tags(
        max_tags: int,
        save_output=True
):
    """
    creates a dictionary with entries for every tag containing lists with games associated with that tag
    :param max_tags: upper bound for most common tags that will be processed
    :param save_output: save result to json file or not
    """
    file_path = "/Volumes/Data/steam/tags"

    # load all files containing each games tags along with the tag count
    files = [f for f in listdir(file_path) if isfile(join(file_path, f)) and f != ".DS_Store"]
    app_ids = [f.split("_")[0] for f in files]
    rev_files = [f"{file_path}/{i}" for i in files]

    # map tags with the associated games
    tag_map = {}
    tqdm_total = len(rev_files)
    for idx, (app_id, rev, file) in tqdm(enumerate(zip(app_ids, rev_files, files)), total=tqdm_total):
        with open(rev, "r") as rev_in:
            tags = ast.literal_eval(rev_in.read())

            # sort list by tag count
            sorted_tags = sorted(tags, key=lambda x: x['count'], reverse=True)

            if len(sorted_tags) < max_tags:
                tag_bound = len(sorted_tags)
            else:
                tag_bound = max_tags

            # only include the most common tags
            for i in range(tag_bound):
                if sorted_tags[i]["name"] in tag_map:
                    tag_map[sorted_tags[i]["name"]].append(file)
                else:
                    tag_map[sorted_tags[i]["name"]] = [file]
    if save_output:
        with open('/Volumes/Data/steam/stats/tags_by_appid.json', 'w') as f:
            json.dump(tag_map, f)


def select_random_review_from_random_game_by_tag_list(
        tag_list: list,
        num_of_reviews_per_tag: int,
        min_token: int,
        max_token: int,
        max_reviews_per_game: int,
        tag_exclusive=True
):
    """
    Function to select random reviews for each tag until a specific number has been selected across all games.

    to-do:
    - count token
    - select review if within token range
    - add to list of reviews
    :param max_reviews_per_game:
    :param tag_list: list of tags
    :param num_of_reviews_per_tag: desired number of selected reviews per tag
    :param min_token: minimum token of a review to have
    :param max_token: maximum token of a review to have
    :param tag_exclusive: boolean if a game should only contain one tag from the list or not
    :return: list of reviews for each tag
    """

    # open dict of all tags with the respective games that have this tag under their most common tags
    with open('/Volumes/Data/steam/stats/tags_by_appid.json', 'r') as f:
        app_ids_by_tag = json.load(f)

    current_tags = tag_list
    # prepare dict for selecting fitting reviews
    selected_reviews = {tag: [] for tag in current_tags}
    # dict for counting how many times a review of a specific game has been selected
    game_count = {tag: {} for tag in current_tags}

    # select fitting reviews until specified number is reached
    while sum(len(rev_list) for rev_list in selected_reviews.values()) < num_of_reviews_per_tag * len(current_tags):
        random_tag = random.choice(current_tags)  # select random tag
        if len(selected_reviews[random_tag]) < num_of_reviews_per_tag:
            filtered_app_ids = app_ids_by_tag[random_tag]  # get all games/appids that have this tag
            random_game = random.choice(filtered_app_ids)  # select random game for a tag

            # check if the selected game only has the selected tag and none of the other tags
            tag_only_once = True
            if tag_exclusive:
                for tag in current_tags:
                    if tag != random_tag:
                        if random_game in app_ids_by_tag[tag]:
                            tag_only_once = False

            # only process further if the check above is true or if exclusivity is disabled
            if tag_exclusive and tag_only_once or not tag_exclusive:
                try:  # try-catch block for file not found errors
                    # open game to get review file
                    with open(f'/Volumes/Data/steam/reviews/{random_game}', 'r') as f:
                        games_reviews = json.load(f)
                    if len(games_reviews["reviews"]) != 0:
                        random_review = random.choice(games_reviews["reviews"])
                        if random_review["language"] == "english":
                            # select random review and count tokens
                            random_review_text = random_review["review"]
                            text = nlp(random_review_text)  # tokenize
                            token_count = len(text)

                            # only process further if token count of review is within desired range
                            if min_token <= token_count <= max_token:
                                if random_review_text not in selected_reviews[random_tag]:
                                    # increment counter for selected game and add to processing list or pass if full
                                    if random_game in game_count[random_tag].keys():
                                        game_count[random_tag][random_game] += 1
                                    else:
                                        game_count[random_tag][random_game] = 1

                                    if game_count[random_tag][random_game] < max_reviews_per_game:
                                        selected_reviews[random_tag].append(random_review_text)
                                    else:
                                        app_ids_by_tag[random_tag].remove(random_game)

                except FileNotFoundError as e:
                    logging.error(str(e))

            # display current amount of collected reviews for monitoring purposes
            current_values = " | ".join([f"{key}: {len(value):0{len(str(num_of_reviews_per_tag))}}"
                                         for key, value in selected_reviews.items()])
            print(f"\r{current_values}", end="")
        else:
            current_tags.remove(random_tag)
    # save collection for further processing (nlp stuff)
    print("")
    print("Collection finished! Saving...")
    with open("/Volumes/Data/steam/finished_corpus/corpus.json", "w") as corpus_out:
        json.dump(selected_reviews, corpus_out)
    with open("/Volumes/Data/steam/finished_corpus/game_count.json", "w") as games_out:
        json.dump(game_count, games_out)


def plot_treemap(data: dict):
    # prepare dimensions
    num_pairs = len(data)
    num_columns = 3
    num_rows = (num_pairs + num_columns - 1) // num_columns

    # fig and subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 6 * num_rows))

    # create treemap for each tag
    for i, (pair, items) in enumerate(data.items()):
        row_idx = i // num_columns
        col_idx = i % num_columns

        labels = ["" for _ in items.values()]  # empty labels
        sizes = list(items.values())

        # generate treemap
        squarify.plot(sizes=sizes, label=labels, ax=axes[row_idx][col_idx], alpha=0.7, edgecolor='white')
        axes[row_idx][col_idx].set_title(f'{pair}')
        axes[row_idx][col_idx].axis('off')

    # save
    plt.tight_layout()
    plt.savefig("/Volumes/Data/steam/finished_corpus/tag_treemap.png", dpi=600)


def plot_distribution(
        data: dict
):
    """
    visualize the number of games that have an above average number of reviews selected to check for possible bias
    :param data: dictionary containing pairs of tags with dicts containing each selected game with the number of
    reviews in the corpus, e.g. {'RPG': {123_Game_title: 5}, ... }
    """
    # filter data to only include games with more than one review
    filtered_data = {tag: [value for value in vals.values() if value > 1] for tag, vals in data.items()}

    # get the highest number of reviews for a single game across all tags (to set as max x value in plot)
    max_x = max(value for values in filtered_data.values() for value in values)

    # reshape data to fit plt requirements
    y = []
    x = range(2, max_x+1)
    for tag, values in filtered_data.items():
        tag_count = []
        for i in x:
            tag_count.append(values.count(i))
        y.append(tag_count)

    sns.set_theme()  # makes everything pretty
    sns.set_context("paper")

    palette = sns.color_palette("Spectral", n_colors=len(filtered_data))

    plt.stackplot(x, y, labels=filtered_data.keys(), colors=palette)

    plt.xlabel("Number of reviews")
    plt.ylabel("Number of games")
    plt.title("Distribution of games with more than one review")
    plt.legend()
    plt.xlim(left=2)  # put graphs directly on left spine

    plt.tight_layout()
    plt.savefig("/Volumes/Data/steam/finished_corpus/tag_distribution.png", dpi=600)
    plt.close()


logging.basicConfig(
    filename='/Volumes/Data/steam/logs/create_corpus.log',
    level=logging.ERROR,
    format='%(asctime)s, %(levelname)s: %(message)s')

nlp = spacy.load("en_core_web_sm")

selected_tags = [
    "Adventure",
    "Strategy",
    "Simulation",
    "RPG",
    "Puzzle"
]

#select_random_review_from_random_game_by_tag_list(selected_tags,50000, 20, 1000, 1000)

with open("/Volumes/Data/steam/finished_corpus/game_count.json", "r") as file_in:
    games = json.load(file_in)

plot_distribution(games)
