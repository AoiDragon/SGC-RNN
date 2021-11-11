import json
import ijson
import numpy as np
from utils import list_add


def preprocess(file_dir, record_dir):
    with open(file_dir, 'r') as f:
        gameRecords = list(ijson.items(f, '', multiple_values=True))

    count = 0
    fileCount = 0
    flag = 0
    for i in range(len(gameRecords)):
        for j in range(len(gameRecords[i])):
            count += 1
            # 简化后的游戏记录
            simplifiedRecord = {}
            record = gameRecords[i][j]
            # 只保留标准角色配置
            if record['roles'] != ['Merlin', 'Percival', 'Assassin', 'Morgana']:
                continue
            # 缺少玩家c信息
            if count == 3875 and not flag:
                flag = 1
                continue
            fileCount += 1
            filename = record_dir + str(fileCount) + '.json'

            # 玩家数量
            simplifiedRecord['numberOfPlayers'] = record['numberOfPlayers']

            # 角色种类
            roles = record['roles']
            roles.insert(0, 'Spy')
            roles.insert(0, 'Resistance')
            simplifiedRecord['roles'] = roles

            # 角色信息用字典保存 key为玩家姓名 value为与roles等长的tensor
            rolesTensor = []
            for player in record['playerRoles'].keys():
                pos = 0
                for role in roles:
                    if record['playerRoles'][player]['role'] == role:
                        rolesTensor.append(pos)
                    pos += 1
            simplifiedRecord['rolesTensor'] = rolesTensor

            # 每种角色人数
            rolesNum = np.zeros(len(roles)).tolist()
            for k in rolesTensor:
                rolesNum[k] += 1
            # for k in range(record['numberOfPlayers']):
            #     player = chr(97+k)
            #     if player == 'a':
            #         rolesNum = rolesTensor[player]
            #     else:
            #         rolesNum = list_add(rolesNum, rolesTensor[player])
            simplifiedRecord['rolesNum'] = rolesNum

            # 任务记录
            History = record['missionHistory']
            missionHistory = []
            for history in History:
                if history == 'succeeded':
                    missionHistory.append(1)
                else:
                    missionHistory.append(0)
            simplifiedRecord['missionHistory'] = missionHistory

            # 对局信息
            gameProcess = []
            voteHistory = record['voteHistory']
            for gameRound in range(len(voteHistory['a'])):
                game = []
                length = len(voteHistory['a'][gameRound])
                for voteRound in range(length):
                    result = []
                    character = []
                    vote = []
                    for k in range(record['numberOfPlayers']):
                        role = chr(97+k)
                        voteResult = voteHistory[role][gameRound][voteRound]

                        if 'VHreject' in voteResult:
                            vote.append(0)
                        elif 'VHapprove':
                            vote.append(1)

                        if 'VHleader' in voteResult:
                            if 'VHpicked' in voteResult:
                                character.append("MemberLeader")
                            else:
                                character.append("nonMemberLeader")
                        elif 'VHpicked' in voteResult and 'VHleader' not in voteResult:
                            character.append('Member')
                        else:
                            character.append('nonMember')
                    result.append(character)
                    result.append(vote)
                    game.append(result)
                gameProcess.append(game)
            simplifiedRecord['gameProcess'] = gameProcess

            with open(filename, 'w') as f:
                f.write(json.dumps(simplifiedRecord, indent=2))

