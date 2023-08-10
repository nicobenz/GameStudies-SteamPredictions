import sqlite3
from os import listdir
from os.path import join, isfile
from tqdm import tqdm
import json

# establish connection
conn = sqlite3.connect('/Volumes/Data/steam/stats/steam_studies.db')
cursor = conn.cursor()

# create db
cursor.execute('''
    CREATE TABLE IF NOT EXISTS games (
        name TEXT NULL,
        id INTEGER PRIMARY KEY,
        reviews INTEGER
    )
''')

review_path = "/Volumes/Data/steam/reviews"
review_files = [f for f in listdir(review_path) if isfile(join(review_path, f)) and f != ".DS_Store"]

# process file names and add game to db
for file in tqdm(review_files):
    with open(f"{review_path}/{file}") as rev_in:
        review = json.loads(rev_in.read())
        db_id = file.split("_")[0]
        db_name = file.replace(".txt", "")
        db_name = db_name.split("_")[1]
        revs = review["reviews"]
        db_count = len(revs)

        cursor.execute('''
                INSERT INTO games (name, id, reviews) VALUES (?, ?, ?)
            ''', (db_name, db_id, db_count))

# commit transaction and close
conn.commit()
conn.close()
