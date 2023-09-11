import subprocess
import time
import re
from bs4 import BeautifulSoup
import requests
import datetime
import json
import os
import random


def save_reviews(app_list, app_index, current_server, servers, cursor, restart):
    app_id = app_list[app_index]
    new_start = restart
    server_count = current_server
    url_game = f"https://store.steampowered.com/app/{app_id}"  # get url of game by app_id
    url_review = f"https://store.steampowered.com/appreviews/{app_id}?json=1"
    game = requests.get(url_game, allow_redirects=False)
    valid = game.status_code  # http code to check for valid url
    if valid == 200:  # http code of 200 means there is a game with this app_id
        soup = BeautifulSoup(game.content, "lxml")
        title = soup.title.text.split(" on Steam")[0]
        title = remove_special_characters(title)

        new_cursor = cursor
        reviews_per_it = 100
        max_reviews = 1000  # high value for initial iteration
        break_count = 0
        review_ids = []
        old_num_reviews = -1
        current_reviews = 0
        while current_reviews < max_reviews:
            if current_reviews == old_num_reviews:
                break_count += 1
            if break_count > 10:
                break
            if current_reviews > 0:
                old_num_reviews = len(review_dict["reviews"])
            if new_cursor is None:
                break
            try:
                payload = {
                    "filter": "all",
                    "language": "all",
                    "cursor": new_cursor.encode(),
                    "purchase_type": "all",
                    "num_per_page": reviews_per_it,
                    "day_range": 9223372036854775807  # for some reason
                }
                review = requests.get(url_review, allow_redirects=False, params=payload)
                if new_cursor == "*":
                    review_dict = review.json()
                    print(f"\nStarting review crawl for game '{title}' ({app_index+1} of {len(app_list)}, "
                          f"{round((app_index/len(app_list))*100, 2)}%) with App ID {app_id}.", end="\n")
                    max_reviews = int(review_dict["query_summary"]["total_reviews"])
                else:
                    for rev in review.json()["reviews"]:
                        if rev["recommendationid"] not in review_ids:
                            break_count = 0
                            review_dict["reviews"].append(rev)
                            review_ids.append(rev["recommendationid"])
                    print(f"\r{len(review_dict['reviews'])} of {max_reviews} reviews, cursor: {new_cursor}",
                          end="", flush=True)
                new_cursor = review.json()["cursor"]
                new_start = False
                current_reviews = len(review_dict["reviews"])
            except:
                server_count += 1
                if server_count >= len(servers):
                    server_count = 0
                with open("data/failed/app_id.txt", "w") as file_out:
                    file_out.write(str(app_id))
                change_server_macos(servers[server_count])
        if current_reviews <= reviews_per_it:
            print(f"\r{len(review_dict['reviews'])} of {max_reviews} reviews, cursor: {new_cursor} ... finished with "
                  f"{max_reviews-len(review_dict['reviews'])} missing at "
                  f"{datetime.datetime.now().strftime('%H:%M:%S')}!")
        else:
            print(f" ... finished with {max_reviews-len(review_dict['reviews'])} missing at "
                  f"{datetime.datetime.now().strftime('%H:%M:%S')}!")
        review_str = json.dumps(review_dict)
        with open(f"data/reviews_full/{app_id}_{title}.txt", "w") as file_out:
            file_out.write(review_str)
        remove_temp()

    with open(f"data/temp/last_id.txt", "w") as file_out:
        file_out.write(f"{app_id}")

    return current_server, new_start


def remove_temp():
    failed_rev = "data/failed/rev.txt"
    failed_cur = "data/failed/cursor.txt"
    failed_app = "data/failed/app_id.txt"
    if os.path.exists(failed_rev):
        os.remove(failed_rev)
    if os.path.exists(failed_cur):
        os.remove(failed_cur)
    if os.path.exists(failed_app):
        os.remove(failed_app)


def change_server_macos(server):
    server = f"{server}.nordvpn.com.tcp"

    disconnect_vpn_macos()
    time.sleep(5)

    applescript = f'''
    tell application "Tunnelblick"
        connect "{server}"
        get state of first configuration where name = "{server}"
        repeat until result = "CONNECTED"
            delay 1
            get state of first configuration where name = "{server}"
        end repeat
    end tell
    '''

    command = ['osascript', '-e', applescript]

    # execute applescript via subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stderr:
        print(f"An error occurred while establishing the VPN connection:\n{stderr.decode()}")
    else:
        print(f"Connected to Server {server} at {datetime.datetime.now().strftime('%H:%M:%S')}")
    time.sleep(5)


def disconnect_vpn_macos():
    applescript = '''
    tell application "Tunnelblick"
    disconnect all
    end tell
    '''

    command = ['osascript', '-e', applescript]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if stderr:
        print(f"An error occurred while executing the AppleScript:\n{stderr.decode()}")
    else:
        pass


def remove_special_characters(string):
    pattern = r'[^a-zA-Z0-9_-]'  # only alphanumeric, - and _
    filtered_string = re.sub(pattern, '', string)
    return filtered_string


server_list = [
    "de688",
    "de848",
    "de947",
    "de1003",
    "de1008",
    "de1009",
    "de1011",
    "de1018",
    "de1020",
    "de1022",
    "de1023",
    "de1025",
    "de1026",
    "de1032",
    "de1033",
    "de1035",
    "de1036",
    "de1039",
    "de1040"
    ]

random.shuffle(server_list)

with open("data/temp/last_id.txt", "r") as file_in:
    app = int(file_in.readline())

with open("data/appid_list.txt", "r") as file_in:
    files = [int(line.rstrip()) for line in file_in]

start_cursor = "*"
is_restart = True

last_index = files.index(app)+1

server_index = 0
change_server_macos(server_list[server_index])

for file_index in range(last_index, 4):
    server_index, is_restart = save_reviews(files, file_index, server_index, server_list, start_cursor, is_restart)
