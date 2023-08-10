"""
Korpus erstellen
Ziele:
- maximale Menge an Daten nutzen
- kein Tag darf übermäßig vertreten sein
- kein Spiel darf übermäßig vertreten sein
- tags: adventure, strategy, simulation, rpg, puzzle, sports
"""

from os import listdir
from os.path import isfile, join
import ast
import json
from tqdm import tqdm
import logging
import random
import spacy


def sort_appid_by_tags(max_tags, save_output=True):
    file_path = "/Volumes/Data/steam/tags"

    files = [f for f in listdir(file_path) if isfile(join(file_path, f)) and f != ".DS_Store"]
    app_ids = [f.split("_")[0] for f in files]
    rev_files = [f"{file_path}/{i}" for i in files]

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

            for i in range(tag_bound):
                if sorted_tags[i]["name"] in tag_map:
                    tag_map[sorted_tags[i]["name"]].append(file)
                else:
                    tag_map[sorted_tags[i]["name"]] = [file]
    if save_output:
        with open('/Volumes/Data/steam/stats/tags_by_appid.json', 'w') as f:
            json.dump(tag_map, f)


def select_random_review_from_random_game_by_tag_list(
        tag_list: list, num_of_reviews_per_tag: int, min_token: int, max_token: int, tag_exclusive=True
):
    """
    Function to select random reviews for each tag until a specific number has been selected across all games.

    to-do:
    - count token
    - select review if within token range
    - add to list of reviews
    :param tag_list: list of tags
    :param num_of_reviews_per_tag: desired number of selected reviews per tag
    :param min_token: minimum token of a review to have
    :param max_token: maximum token of a review to have
    :param tag_exclusive: boolean if a game should only contain one tag from the list or not
    :return: list of reviews for each tag
    """

    with open('/Volumes/Data/steam/stats/tags_by_appid.json', 'r') as f:
        app_ids_by_tag = json.load(f)

    selected_reviews = [[0] for _ in tag_list]

    # select fitting reviews until specified number is reached
    while sum(sum(sublist) for sublist in selected_reviews) < num_of_reviews_per_tag * len(tag_list):
        random_tag = random.choice(tag_list)
        filtered_app_ids = app_ids_by_tag[random_tag]
        random_game = random.choice(filtered_app_ids)

        tag_only_once = True
        if tag_exclusive:
            for tag in tag_list:
                if tag == random_tag:
                    pass
                else:
                    if random_game in app_ids_by_tag[tag]:
                        tag_only_once = False
                    else:
                        pass
        else:
            pass

        if tag_exclusive and tag_only_once or not tag_exclusive:
            try:
                with open(f'/Volumes/Data/steam/reviews/{random_game}', 'r') as f:
                    games_reviews = json.load(f)
                if len(games_reviews["reviews"]) == 0:
                    pass
                else:
                    random_review = random.choice(games_reviews["reviews"])
                    if random_review["language"] != "english":
                        pass
                    else:
                        random_review_text = random_review["review"]
                        print(random_review_text)

            except FileNotFoundError as e:
                logging.error(str(e))
        else:
            pass


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
    "Puzzle",
    "Sports"
]

select_random_review_from_random_game_by_tag_list(selected_tags, 10000, 20, 1000)
