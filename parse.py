from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
import ast


def process_review(data):
    # do stuff
    pass


def parse_html():
    src_path = "data/source/"
    src_files = [
        f for f in listdir(src_path) if isfile(join(src_path, f)) and "DS_Store" not in f
    ]  # get all files from source folder and filter .DS_Store which is a hidden macOS file
    rev_path = "data/reviews/"
    rev_files = [f for f in listdir(rev_path) if isfile(join(rev_path, f)) and "DS_Store" not in f]

    for src, rev in zip(src_files, rev_files):
        with open(f"{src_path}/{src}", "r") as html:
            soup = BeautifulSoup(html, "lxml")
        with open(f"{rev_path}/{rev}", "r") as review:
            print(rev)
            review_content = review.read()
            review_content = ast.literal_eval(review_content)

        title = soup.title.text.split(" on Steam")[0]
        tags = soup.find_all("a", {"class": "app_tag"})
        tag_list = [tag.text.strip() for tag in tags]
        review_data = {
            "title": title,
            "tags": tag_list,
            "metadata": review_content["query_summary"],
            "reviews": review_content["reviews"]
        }

        process_review(review_data)


parse_html()
