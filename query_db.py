import sqlite3

# establish connection
conn = sqlite3.connect('/Volumes/Data/steam/stats/steam_studies.db')
cursor = conn.cursor()

# query
cursor.execute('SELECT COUNT(name) FROM games WHERE reviews BETWEEN 10000 and 100000')

# get result
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
