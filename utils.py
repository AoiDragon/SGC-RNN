import torch
import numpy as np


def generate_graph(record):
    """
    Generate player graph for every game.
    :param record: Game record.
    :return edges: Graph list.
    """
    graphs = []                             # 存储一次游戏中所有图的列表
    game = record['gameProcess']            # 当前游戏
    playerNum = record["numberOfPlayers"]   # 玩家人数
    for mission in game:                    # 当前任务
        missionGraphs = []                  # 存储一次任务中所有图的列表
        for vote in mission:                # 当前投票
            role = vote[0]                  # 玩家角色
            voteResult = vote[1]            # 投票结果
            # 存储当前投票结果对应的图的字典
            voteGraph = {"numberOfPlayers": record["numberOfPlayers"], "positiveEdges": [], "negativeEdges": [],
                         "Members": [], "nonMembers": []}
            for k in range(playerNum):
                player = role[k]
                if player == "Member":  # 之后改一下数据集，让变量命名一致
                    voteGraph["Members"].append(k)
                elif player == "nonMember":
                    voteGraph["nonMembers"].append(k)
                else:
                    voteGraph["Leader"] = k
                    if player == "MemberLeader":
                        voteGraph["Members"].append(k)
                    else:
                        voteGraph["nonMembers"].append(k)

            for k in range(playerNum):
                for m in range(playerNum):
                    if k == m:
                        continue
                    elif voteResult[k] == voteResult[m]:
                        voteGraph["positiveEdges"].append([k, m])
                    else:
                        voteGraph["negativeEdges"].append([k, m])
            missionGraphs.append(voteGraph)
        graphs.append(missionGraphs)
    return graphs


def generate_embedding(size):
    """
    Generate initial embedding.
    :param size: Size of embedding.
    :return: Randomly generated embedding.
    """
    x = torch.randn(size, 1)  # 标准正态分布
    X = x.numpy()
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # 归一化到[0,1]
    return X_std
