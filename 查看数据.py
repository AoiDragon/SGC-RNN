import os
import json


for root, dirs, files in os.walk('./data/fullGameRecord/'):
    cnt = 0
    for file in files:
        cnt += 1
        filename = './data/fullGameRecord/' + str(cnt) + '.json'
        with open(filename, 'r', encoding="utf8") as f:
            gameData = json.loads(f.read())
            if gameData['numberOfPlayers'] == 10:
                print(cnt)