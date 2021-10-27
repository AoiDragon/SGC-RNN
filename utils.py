import torch
import numpy as np
import scipy.stats as stats


def generate_graph(records):
    """
    Generate player graph for every game.
    :param record: Game record.
    :return edges: Graph list.
    """
    G = []                                      # 存储一个batch中所有游戏的图
    for record in records:
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
        G.append(graphs)
    return G


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


def initialize_embedding(self):
    """
    生成玩家的初始嵌入。服从均值为0.5，标准差为1，分布在[0,1]的截断正态分布
    :return:
    """
    mu, sigma = 0.5, 1
    lower, upper = 0, 1
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 生成分布
    h = X.rvs(7)  # 取样
    h = torch.from_numpy(h)
    return h