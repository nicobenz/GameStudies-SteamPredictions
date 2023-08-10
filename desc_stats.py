from os import listdir
from os.path import isfile, join
import json
import ast
from tqdm import tqdm
import re


def count_tags_for_tags(num_of_tags=0):
    with open("/Volumes/Data/steam/stats/all_tags.txt", "r") as f:
        tags = f.read()
        tags = ast.literal_eval(tags)
        tag_names = list(tags.keys())
    if num_of_tags > 0:
        tag_names = tag_names[:num_of_tags]
    tag_collection = {}
    for tag in tag_names:
        collect = {}
        for t in tag_names:
            collect[t] = 0
        tag_collection[tag] = collect

    for file in tqdm(rev_files):
        with open(f"{rev_path}/{file}", "r") as f:
            read_dict = json.loads(f.read())
            rev_files.append(read_dict)
        if file in tag_files:
            with open(f"{tag_path}/{file}", "r") as f:
                read_tags = ast.literal_eval(f.read())
                for tag in read_tags:
                    for t in read_tags:
                        if tag["name"] in tag_collection:
                            if t["name"] in tag_collection[tag["name"]]:
                                if t["name"] != tag["name"]:
                                    tag_collection[tag["name"]][t["name"]] += 1
    with open("/Volumes/Data/steam/stats/tag_overlap.txt", "w") as f:
        f.write(str(tag_collection))

    for tag_k, tag_v in tag_collection:
        for k, v in tag_v:
            print(f"{tag_k} -> {k}: {v}")
        print("")


def count_tags():
    tag_count = {}
    # load reviews in list
    for file in tqdm(rev_files):
        with open(f"{rev_path}/{file}", "r") as f:
            read_dict = json.loads(f.read())
            rev_files.append(read_dict)
            num_revs = len(read_dict["reviews"])
        if file in tag_files:
            with open(f"{tag_path}/{file}", "r") as f:
                read_tags = ast.literal_eval(f.read())
                for tag in read_tags:
                    if tag["name"] in tag_count:
                        tag_count[tag["name"]] += num_revs
                    else:
                        tag_count[tag["name"]] = num_revs


def count_token(path, all_reviews):
    token_count = 0
    for review in tqdm(all_reviews):
        with open(f"{path}/{review}") as f:
            reviews = json.loads(f.read())
            if len(reviews["reviews"]) > 0:
                for rev in reviews["reviews"]:
                    if rev["language"] == "english":
                        if rev["review"] is not None:
                            text = re.sub(r'[^a-zA-Z]', ' ', rev["review"])
                            text = ' '.join(text.split())
                            token_count += len(text)
    return token_count


rev_path = "/Volumes/Data/steam/reviews"
tag_path = "/Volumes/Data/steam/tags"

rev_files = [f for f in listdir(rev_path) if isfile(join(rev_path, f)) and f != ".DS_Store"]
tag_files = [f for f in listdir(tag_path) if isfile(join(tag_path, f)) and f != ".DS_Store"]

all_token = count_token(rev_path, rev_files)

with open("/Volumes/Data/steam/stats/all_token.txt", "w") as f:
    f.write(str(all_token))
