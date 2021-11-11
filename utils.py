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
        playerNum = record["numberOfPlayers"]  # 玩家人数
        missionCount = 0
        for mission in game:  # 当前任务
            missionGraphs = []  # 存储一次任务中所有图的列表
            for vote in mission:  # 当前投票
                role = vote[0]  # 玩家角色
                voteResult = vote[1]  # 投票结果
                # 存储当前投票结果对应的图的字典
                voteGraph = {"numberOfPlayers": record["numberOfPlayers"], "positiveEdges": [], "negativeEdges": [],
                             "Members": [], "nonMembers": [], "Leader": [], "missionResult": record["missionHistory"][missionCount],
                             "rolesNum": record["rolesNum"]}
                for k in range(playerNum):
                    player = role[k]
                    if player == "Member":  # 之后改一下数据集，让变量命名一致
                        voteGraph["Members"].append(k)
                    elif player == "nonMember":
                        voteGraph["nonMembers"].append(k)
                    else:
                        voteGraph["Leader"].append(k)
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
    h = X.rvs(size)  # 取样
    h = torch.from_numpy(h)
    return h


def padding(embedding_list, p_size, e_size, device):
    """
    对sgcn的输出结果进行padding
    :param device: cpu或cuda
    :param p_size: 填充后一共有多少个tensor
    :param e_size: 每个tensor的大小
    :param embedding_list: 已有embedding的列表
    :return:
    """
    while len(embedding_list) < 25:
        padding_tensor = torch.full([e_size], -1).float().to(device)
        embedding_list.append(padding_tensor)
    # embedding = embedding_list[0]
    # for i in range(1, p_size):
    #     # 拼接已有tensor
    #     if i < len(embedding_list):
    #         embedding = torch.cat((embedding, embedding_list[i]), 0)
    #     # padding
    #     else:
    #         padding_tensor = torch.full((1, e_size), -1)
    #         embedding = torch.cat((embedding, padding_tensor), 0)
    return embedding_list


def judge(h):
    """
    判断一个tensor是不是补充的（全是-1）
    :param h: 待判断的tensor
    :return:
    """
    for i in range(h.size()[1]):
        if h[0, i] != -1:
            return False
        return True
    # for h_i in h:
    #     if h_i != -1:
    #         return False
    #     return True


def list_add(x, y):
    """
    列表相加
    :param x: 列表1
    :param y: 列表2
    :return:
    """
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    return z


def compute_rank(h):
    h = h.numpy()
    x = np.argsort(-h)
    y = np.zeros(len(x))
    rank = 1
    for i in x:
        y[i] = rank
        rank += 1
    return y
