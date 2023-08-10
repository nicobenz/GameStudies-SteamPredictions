import json
import requests

# get current list of games present on steam
all_apps_url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
apps_url = requests.get(all_apps_url)
all_apps = json.loads(apps_url.content)
app_list = all_apps["applist"]["apps"]

# get only app ids for later crawling
apps = []
for app in app_list:
    apps.append(int(app["appid"]))

apps.sort()

# save
with open("data/appid_list.txt", "w") as f_out:
    for file in apps:
        f_out.write(f"{file}\n")
