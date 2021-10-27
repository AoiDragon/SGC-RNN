import torch
import numpy as np
import scipy.stats as stats


def generate_graph(records):
    """
    Generate player graph for every game.
    :param records: Game record.
    :return edges: Graph list.
    """
    G = []  # 存储一个batch中所有游戏的图
    for record in records:
        graphs = []  # 存储一次游戏中所有图的列表
        game = record['gameProcess']  # 当前游戏
        print(record['missionHistory'])
        playerNum = record["numberOfPlayers"]  # 玩家人数
        missionCount = 0
        for mission in game:  # 当前任务
            missionGraphs = []  # 存储一次任务中所有图的列表
            for vote in mission:  # 当前投票
                role = vote[0]  # 玩家角色
                voteResult = vote[1]  # 投票结果
                # 存储当前投票结果对应的图的字典
                voteGraph = {"numberOfPlayers": record["numberOfPlayers"], "positiveEdges": [], "negativeEdges": [],
                             "Members": [], "nonMembers": [], "missionResult": record['missionHistory'][missionCount]}
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
            missionCount += 1
            graphs.append(missionGraphs)
        print(graphs)
        G.append(graphs)
    return G


def initialize_embedding(size):
    """
    生成玩家的初始嵌入。服从均值为0.5，标准差为1，分布在[0,1]的截断正态分布
    :return: 形状为1*7的tensor
    """
    mu, sigma = 0.5, 1
    lower, upper = 0, 1
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 生成分布
    h = X.rvs((1, size))  # 取样
    h = torch.from_numpy(h)
    return h


def padding(p_size, e_size, mission_embedding_list):
    """
    对sgcn的输出结果进行padding
    :param p_size: 填充后一共有多少个tensor
    :param e_size: 每个tensor的大小
    :param mission_embedding_list: 已有embedding的列表
    :return:
    """
    mission_embedding = mission_embedding_list[0]
    for i in range(1, p_size):
        # 拼接已有tensor
        if i < len(mission_embedding_list):
            mission_embedding = torch.cat((mission_embedding, mission_embedding_list[i]), 0)
        # padding
        else:
            padding_tensor = torch.full((1, e_size), -1)
            mission_embedding = torch.cat((mission_embedding, padding_tensor), 0)
    return mission_embedding


def judge(h):
    """
    判断一个tensor是不是补充的（全是-1）
    :param h: 待判断的tensor
    :return:
    """
    for h_i in h:
        if h_i != -1:
            return False
        return True
