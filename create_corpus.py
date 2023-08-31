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
import sqlite3
from tqdm import tqdm
import logging
import random
import spacy
import matplotlib.pyplot as plt
import squarify
import seaborn as sns
import multiprocessing
import itertools
import time
import re
from send_notification import send_mail, format_time
from datetime import datetime


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
        tag_lists: list,
        num_of_reviews_per_tag: int,
        min_token: int,
        max_token: int,
        max_reviews_per_game: int,
        path: str,
        tag_exclusive=True
):
    """
    Function to select random reviews for each tag until a specific number has been selected across all games.

    to-do:
    - count token
    - select review if within token range
    - add to list of reviews
    :param max_reviews_per_game:
    :param tag_lists: list of lists of tags
    :param num_of_reviews_per_tag: desired number of selected reviews per tag
    :param min_token: minimum token of a review to have
    :param max_token: maximum token of a review to have
    :param tag_exclusive: boolean if a game should only contain one tag from the list or not
    :return: list of reviews for each tag
    """
    for idx, tag_list in enumerate(tag_lists, start=1):
        # open dict of all tags with the respective games that have this tag under their most common tags
        with open(f'{path}/Volumes/Data/steam/stats/tags_by_appid.json', 'r') as f:
            app_ids_by_tag = json.load(f)

        # prepare dict for selecting fitting reviews
        pool = multiprocessing.Pool(processes=len(tag_list))

        # List of tags to process
        tags_to_process = tag_list.copy()

        # params: tag, max_reviews, filtered_app_ids, all_tags, all_games, min_token, max_token, max_revs_per_game, path
        params = [(tag, num_of_reviews_per_tag, app_ids_by_tag[tag], tag_list, app_ids_by_tag, min_token, max_token,
                  max_reviews_per_game, path) for tag in tags_to_process]

        print("Starting collection.")

        params = [list(param) for param in params]

        start_time = time.time()

        # hand function to threads
        results = pool.map(process_tag, params)

        # pool needs to be closed for threads to start
        pool.close()

        # wait for every thread to finish
        pool.join()

        # split thread results into both categories (tokens and game count)
        tokens = []
        game_count = []
        for res in results:
            for num, r in enumerate(res):
                if num == 0:
                    tokens.append(r)
                else:
                    game_count.append(r)

        # flatten both (from nested tags to flat tags)
        tokens_dict = {}
        for tag, tok in zip(tags_to_process, tokens):
            tokens_dict[tag] = tok
        game_count_dict = {}
        for count in game_count:
            for k, v in count.items():
                game_count_dict[k] = v

        # save for later
        with (open(f"{path}/Volumes/Data/steam/finished_corpus/corpora/corpus-{idx}-{''.join(tag_list)}.json", "w")
              as tokens_out):
            json.dump(tokens_dict, tokens_out)
        with (open(f"{path}/Volumes/Data/steam/finished_corpus/game_counts/corpus-{idx}-{''.join(tag_list)}.json", "w")
              as tokens_out):
            json.dump(game_count_dict, tokens_out)

        # calculate and format run time for notification service
        end_time = time.time()
        time_taken = end_time - start_time
        time_taken = format_time(time_taken)
        now = datetime.now().strftime('%H:%M')
        collected = []
        for k, v in tokens_dict.items():
            collected.append(f"{k}: {len(v)} revs")

        msg = (f"Saved corpus {idx} of {len(tag_lists)} '{''.join(tag_list)}' after {time_taken} at {now}.\n\n"
               f"Data collected:\n{', '.join(collected)}")
        send_mail("nico-benz@gmx.net", msg)


def select_game_path(game, ids):
    game = game.split("_")[0]

    if game in ids:
        items = len(ids[game])
        if items == 1:
            return ids[game][0]
        elif items > 1:
            return random.choice(ids[game])
    else:
        return False


def process_tag(parameters: list):
    tag, max_reviews, filtered_app_ids, all_tags, all_games, min_token, max_token, max_reviews_per_game, path = parameters
    tag_exclusive = True
    review_tokens = []
    selected_reviews = []
    game_count = {}
    nlp = spacy.load("en_core_web_md")

    if max_reviews >= 500:
        percentage_steps = 1
        progress_list = [x for x in range(int(max_reviews / (100 / percentage_steps)),
                                          max_reviews + 1, int(max_reviews / (100 / percentage_steps)))]
    else:
        progress_list = []
    all_appids = [file for file in listdir(f"{path}/Volumes/Data/steam/reviews") if ".DS_Store" not in file]
    app_id_mapping = {}
    for file in all_appids:
        app_id = file.split("_")[0]
        if app_id in app_id_mapping:
            app_id_mapping[app_id].append(file)
        else:
            app_id_mapping[app_id] = [file]

    while len(review_tokens) < max_reviews:
        random_game = random.choice(filtered_app_ids)  # select random game for a tag
        random_game = select_game_path(random_game, app_id_mapping)
        # check if the selected game only has the selected tag and none of the other tags
        if random_game:
            tag_only_once = True
            if tag_exclusive:
                for t in all_tags:
                    if t != tag:
                        if random_game in all_games[t]:
                            tag_only_once = False

            # only process further if the check above is true or if exclusivity is disabled
            if tag_exclusive and tag_only_once or not tag_exclusive:
                try:  # try-catch block for file not found errors
                    # open game to get review file
                    with open(f'{path}/Volumes/Data/steam/reviews/{random_game}', 'r') as f:
                        games_reviews = json.load(f)
                    if len(games_reviews["reviews"]) > 0:
                        random_review = random.choice(games_reviews["reviews"])
                        if random_review["language"] == "english":
                            # select random review and count tokens
                            random_review_text = random_review["review"]
                            if min_token <= len(random_review_text) <= max_token:
                                cleaned_text = clean_text(random_review_text, nlp)
                                token_count = len(cleaned_text)
                                # only process further if token count of review is within desired range
                                if min_token <= token_count <= max_token:
                                    if random_review["recommendationid"] not in selected_reviews:
                                        # increment counter for selected game and add to processing list or pass if full
                                        selected_reviews.append(random_review["recommendationid"])
                                        review_tokens.append(cleaned_text)

                                        if random_game in game_count.keys():
                                            game_count[random_game] += 1
                                        else:
                                            game_count[random_game] = 1

                                        if game_count[random_game] >= max_reviews_per_game:
                                            filtered_app_ids.remove(random_game)

                                        # progress information
                                        if progress_list:
                                            current_progress = len(review_tokens)
                                            if current_progress in progress_list:
                                                print(f"{int((current_progress / max_reviews) * 100):8}%: {tag}")
                except Exception as e:
                    print(e)
    print(f"Finished: {tag}")
    return [review_tokens, game_count]


def clean_text(text, nlp):
    # tokenize, lowercase, remove stopwords, and lemmatize
    custom_stops = remove_custom_stopwords()  # based on tf-idf values from earlier corpora where these ranked high
    doc = nlp(remove_ascii_art(text))
    cleaned_tokens = [
        token.text.lower() for token in doc
        if token.is_alpha and not token.is_stop and token.lemma_ not in custom_stops
    ]
    return cleaned_tokens


def remove_ascii_art(text):
    ascii_removed = re.sub(r'[^\x00-\x7F]+', '', text)
    return ascii_removed


def remove_custom_stopwords():
    custom_stop_words = [
        "game",
        "like",
        "good",
        "games",
        "time",
        "play",
        "fun",
        "way",
        "great",
        "little",
        "bit",
        "lot",
        "pretty",
        "feel",
        "think",
        "recommend",
        "playing",
        "things",
        "want",
        "different",
        "played",
        "worth",
        "got",
        "love",
        "better",
        "new",
        "need",
        "find",
        "bad",
        "nice",
        "steam",
        "know",
        "dlc",
        "use",
        "hours",
        "people"
    ]
    return custom_stop_words


def remove_special_characters(text, nlp):
    doc = nlp(text)
    cleaned_tokens = [token.text for token in doc if token.is_alpha]
    cleaned_text = " ".join(cleaned_tokens)
    return cleaned_text


def remove_stopwords(text, nlp):
    doc = nlp(text)
    cleaned_tokens = [token.text for token in doc if not token.is_stop]
    cleaned_text = " ".join(cleaned_tokens)
    return cleaned_text


def remove_named_entities(text, nlp):
    doc = nlp(text)
    tokens_without_entities = [token.text if not token.ent_type_ else '' for token in doc]
    cleaned_text = " ".join(tokens_without_entities).strip()
    return cleaned_text


def lemmatise(text, nlp):
    doc = nlp(text)
    cleaned_tokens = [token.lemma_ for token in doc]
    cleaned_text = " ".join(cleaned_tokens)
    return cleaned_text


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


def plot_distribution(data: dict):
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


def create_full_corpus(max_reviews_per_game, min_token, max_token):
    game_path = "/Volumes/Data/steam/reviews"
    all_games = [x for x in listdir("/Volumes/Data/steam/reviews") if ".DS_Store" not in x]

    games_dict = {}
    for game in all_games:
        with open(f"{game_path}/{game}") as file_in:
            game_file = file_in.read()

        print(game_file)
        game_infos = {game: {}}

    # continue coding here


def create_flat_db_corpus(most_common_tags=10):
    # prepare db
    conn = sqlite3.connect("/Volumes/Data/steam/finished_corpus/reviews.db")
    cursor = conn.cursor()

    # create table if not existing
    create_db = """
    CREATE TABLE IF NOT EXISTS reviews (
        app_id INTEGER,
        rev_id INTEGER PRIMARY KEY,
        sorted_tags TEXT,
        voted_up BOOLEAN,
        votes_up INTEGER,
        weighted_vote_score REAL,
        review TEXT
    );
    """
    cursor.execute(create_db)
    conn.commit()

    missing_tags = []  # download missing files later
    game_path = "/Volumes/Data/steam/reviews"
    tags_path = "/Volumes/Data/steam/tags"
    all_games = [x for x in listdir("/Volumes/Data/steam/reviews") if ".DS_Store" not in x]

    # create a corpus only containing review dicts to eliminate the need to open files every time
    for game in tqdm(all_games, desc="Games"):
        with open(f"{game_path}/{game}", "r") as file_in:
            game_file = json.loads(file_in.read())

        app_id = game.split("_")[0]
        if game_file["reviews"]:
            # loop over all reviews
            for rev in game_file["reviews"]:
                if rev["language"] == "english":
                    try:
                        # load all tags
                        with open(f"{tags_path}/{game}", "r") as file_in:
                            tags_file = ast.literal_eval(file_in.read())

                        num_of_tags = len(tags_file)
                        tags = []
                        # only use up to the specified amount of tags
                        if num_of_tags < most_common_tags:
                            for i in range(len(tags_file)):
                                tag_dict = {"name": tags_file[i]["name"], "count": tags_file[i]["count"]}
                                tags.append(tag_dict)
                        else:
                            for i in range(most_common_tags):
                                tag_dict = {"name": tags_file[i]["name"], "count": tags_file[i]["count"]}
                                tags.append(tag_dict)

                        # bundle up dict with everything that could be useful
                        tags_json = json.dumps(tags)
                        # make rev id more unique in case of duplicates
                        better_rev_id = int(f"{rev['recommendationid']}{random.randint(1000, 9999)}")
                        review_data = {
                            "app_id": app_id,
                            "rev_id": better_rev_id,
                            "sorted_tags": tags_json,
                            "voted_up": rev["voted_up"],
                            "votes_up": rev["votes_up"],
                            "weighted_vote_score": rev["weighted_vote_score"],
                            "review": rev["review"]
                        }

                        # add to corpus under review id
                        try:
                            insert_query = """
                            INSERT INTO reviews
                            (app_id, rev_id, sorted_tags, voted_up, votes_up, weighted_vote_score, review)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """
                            cursor.execute(insert_query, (
                                review_data["app_id"],
                                review_data["rev_id"],
                                review_data["sorted_tags"],
                                review_data["voted_up"],
                                review_data["votes_up"],
                                review_data["weighted_vote_score"],
                                review_data["review"]
                            ))
                            conn.commit()
                        except sqlite3.IntegrityError as e:
                            print(f"Error adding record {review_data['rev_id']} of '{game}' to DB:", e)
                    except FileNotFoundError:
                        # add to missing list if file not found
                        missing_tags.append(game)
    # save
    with open("/Volumes/Data/steam/stats/missing_tags.txt", "w") as file_out:
        for line in missing_tags:
            file_out.write(f"{line}\n")


def process_db(tag_num, max_games, min_reviews_per_game, min_token, max_token):
    conn = sqlite3.connect('/Volumes/Data/steam/finished_corpus/reviews.db')
    cursor = conn.cursor()

    #cursor.execute("SELECT COUNT(*) FROM reviews")
    #num_rows = cursor.fetchone()[0]
    num_rows = 10

    initial_order = [x for x in range(1, num_rows+1)]
    random.shuffle(initial_order)

    corpus_data = {"reviews": [], "labels": [], "weights": []}
    app_id_counter = {}
    for i in tqdm(initial_order):
        cursor.execute("SELECT review, app_id, sorted_tags FROM reviews WHERE row_index = ?", (i,))
        result = cursor.fetchall()
        app_id = result[0][1]

        # only proceed if count of game under specified value
        if app_id not in app_id_counter or app_id in app_id_counter and app_id_counter[app_id] < max_games:
            tag_list = ast.literal_eval(result[0][2])
            if tag_list:
                # process review
                review = result[0][0]
                nlp = spacy.load("en_core_web_md")
                # do more nlp stuff here:
                # casing, tokenizing, removing of special characters, numbers, ascii art and stop words, lemmatizing

                # filter tag list
                tag_count = tag_num if tag_num < len(tag_list) else len(tag_list)
                tag_list = [tag_list[i] for i in range(tag_count)]

                # process labels and weights
                labels = [tag["name"] for tag in tag_list]
                weights = [(tag["count"] / sum(d['count'] for d in tag_list)) for tag in tag_list]

                # add everything to collection
                corpus_data["reviews"].append(review)
                corpus_data["labels"].append(labels)
                corpus_data["weights"].append(weights)

                # update app_id counter
                if app_id in app_id_counter:
                    app_id_counter[app_id] += 1
                else:
                    app_id_counter[app_id] = 1

    with open("/Volumes/Data/steam/finished_corpus/reviews.json", "w") as file_out:
        json.dump(corpus_data, file_out)

    conn.close()


if __name__ == '__main__':
    # set main working directory here (to use code on nas with mirrored data in different dir)
    # set to empty string when working with external drive mounted to local machine
    # set to "/volume1/Cloud/code/steam_corpus" when working on synology nas
    volume_path = ""

    # error logger
    logging.basicConfig(
        filename=f'{volume_path}/Volumes/Data/steam/logs/corpus_creation_errors.log',
        level=logging.INFO,
        format='%(asctime)s, %(levelname)s: %(message)s')

    # most common tags in raw corpus
    most_common_tags = [
        "Indie", "Action", "Casual", "Adventure", "Strategy",
        "Simulation", "RPG", "FreetoPlay", "Puzzle", "EarlyAccess",
        "SexualContent", "Nudity", "Racing", "Sports", "VisualNovel"
    ]

    # tags i want to create a corpus with
    selected_tags = [
        ["Strategy", "Simulation", "RPG", "Puzzle", "Sports"],
        ["Indie", "Action", "Casual", "Adventure", "Strategy"],
        random.sample(most_common_tags, 5),
        random.sample(most_common_tags, 5),
        random.sample(most_common_tags, 5)
    ]

    # create a corpus with given parameters
    select_random_review_from_random_game_by_tag_list(
        selected_tags,
        50000,
        20,
        1000,
        1000,
        volume_path
    )

    #with open("/Volumes/Data/steam/finished_corpus/game_count.json", "r") as file_in:
    #    games = json.load(file_in)

    #plot_distribution(games)
    #create_full_corpus(1000, 20, 1000)
    #create_flat_db_corpus()
    #process_db(5, 1000, 10, 20, 1000)

