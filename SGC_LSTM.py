import torch
import numpy as np
from signed_sage_convolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep, ListModule
from utils import initialize_embedding, padding, judge
import time


class SGC_LSTM(torch.nn.Module):
    """
    SGC_LSTM NetWork Class.
    """

    def __init__(self, device, config):
        """
        Initialize SGC_LSTM.
        :param device: Device for calculations.
        :param config: Arguments object.
        """
        super(SGC_LSTM, self).__init__()
        # 参数对象
        self.device = device
        self.config = config
        self.cell_num = self.config.cell_num
        self.lstm = []

        self.setup_layers()

    def setup_sgcn(self):
        """
        搭建sgcn
        """
        # 输入是28*1（7*4，3种一维邻居的聚合结果7，一共21，自己7，拼接后28）的向量
        self.positive_base_aggregator = SignedSAGEConvolutionBase(self.config.embedding_size * 4, 32,
                                                                  device=self.device).to(self.device)
        self.negative_base_aggregator = SignedSAGEConvolutionBase(self.config.embedding_size * 4, 32,
                                                                  device=self.device).to(self.device)

        self.positive_aggregators = []
        self.negative_aggregators = []
        for i in range(2):
            # 输入是32*7（6种+自己）输出暂定32*1
            self.positive_aggregators.append(SignedSAGEConvolutionDeep(32 * 7, 32, device=self.device).to(self.device))

            self.negative_aggregators.append(SignedSAGEConvolutionDeep(32 * 7, 32, device=self.device).to(self.device))

        self.positive_aggregators = ListModule(*self.positive_aggregators)
        self.negative_aggregators = ListModule(*self.negative_aggregators)

    def setup_layers(self):
        # setup_sgcn只能生成满足一轮任务的sgcn，应该调用和游戏轮数相同次，每个游戏用的不一样
        # sgcn中未包含激活函数
        self.setup_sgcn()
        for _ in range(self.cell_num):
            self.lstm.append(torch.nn.LSTMCell(self.config.lstm_input_size, self.config.lstm_hidden_size))
        self.lstm_list = ListModule(*self.lstm)
        self.W = torch.nn.Linear(self.config.lstm_hidden_size, self.config.embedding_size)

    def forward(self, graphs):
        """

        :param graphs:
        :return:
        """
        # 先串行进行SGCN计算
        # game是单独一局游戏
        out = []
        for game in graphs:
            player_num = game[0][0]["numberOfPlayers"]
            embedding_size = self.config.embedding_size
            game_embedding, game_embedding_list = [], []

            for _ in range(player_num):
                game_embedding_list.append([])

            for mission in game:
                add_info = []  # 存储每个玩家的任务额外信息（任务成功，是否参与任务）
                mission_embedding_list = []  # 临时存储任务中每一轮投票的嵌入,二维列表，第一维为角色

                for _ in range(player_num):
                    add_info.append(torch.ones(2).float().to(self.device))
                    mission_embedding_list.append([])

                for vote in mission:
                    # 生成初始嵌入
                    h_0 = []  # 初始嵌入
                    for _ in range(player_num):
                        mission_embedding_list.append([])
                        h_0.append(initialize_embedding(embedding_size).float().to(self.device))
                    # 目前sgcn接收一轮投票的图，返回所有玩家当前的嵌入
                    # 因此这里还需要加一个玩家的维度
                    # h_pos[i]表示第i次聚合后所有点的嵌入的列表
                    h_pos, h_neg = [], []
                    # 进行第一层SGCN
                    h_pos.append(self.positive_base_aggregator(vote, "positive", h_0))
                    h_neg.append(self.negative_base_aggregator(vote, "negative", h_0))

                    # 第二层SGCN
                    for i in range(1, self.config.layer_num):
                        h_pos.append(
                            self.positive_aggregators[i - 1](vote, "positive", h_pos[i - 1], h_neg[i - 1]))
                        h_neg.append(
                            self.negative_aggregators[i - 1](vote, "negative", h_pos[i - 1], h_neg[i - 1]))
                    # h_pos[i] player_num个1*32tensor

                    for player in range(player_num):
                        mission_embedding_list[player].append(
                            torch.cat((h_pos[-1][player], h_neg[-1][player]), 0))  # player_num个1*64tensor
                        # 修改add_info
                        if player not in vote["nonMembers"]:
                            add_info[player][1] = 0  # 任意一轮投票中未参与组队，修改add_info
                        add_info[player][0] = vote["missionResult"]  # 添加任务结果，每个角色只用只添加一次

                # 对当前任务进行padding（填充一个全为-1的向量），获得一个5*66(输出64+额外添加的2)的向量
                for player in range(player_num):
                    #  添加补充信息
                    for i in range(len(mission_embedding_list[player])):
                        mission_embedding_list[player][i] = torch.cat(
                            (mission_embedding_list[player][i], add_info[player]), 0)
                    # padding
                    for i in range(5):
                        # 有对应的投票，直接复制
                        if i < len(mission_embedding_list[player]):
                            game_embedding_list[player].append(mission_embedding_list[player][i])
                        # 否则添加全为-1的tensor
                        else:
                            game_embedding_list[player].append(torch.full([self.config.lstm_input_size], -1)
                                                               .float().to(self.device))

            h_final = []  # 最终嵌入的列表
            for player in range(player_num):
                embedding = padding(game_embedding_list[player], 25,
                                    self.config.lstm_input_size, self.device)  # 一个玩家在该轮任务中的嵌入，形状为25*66
                hx = torch.FloatTensor(1, self.config.lstm_hidden_size).uniform_().to(self.device)  # 初始化方式待讨论
                cx = torch.FloatTensor(1, self.config.lstm_hidden_size).uniform_().to(self.device)
                for i in range(len(embedding)):
                    embedding[i] = torch.unsqueeze(embedding[i], dim=0)
                for i in range(self.cell_num):
                    if not judge(embedding[i]):  # 判断是不是全是-1
                        cell = self.lstm_list[i]
                        hx, cx = cell(embedding[i], (hx, cx))  # 待讨论
                h_final.append(self.W(hx))

            h_f = h_final[0]
            for k in range(1, len(h_final)):
                h_f = torch.cat((h_f, h_final[k]), 0)
            out.append(h_f)
            # 看一下torch的自动求梯度原理，是否需要一起求loss还是说可以返回后分开求。
            # h_out = []
            # # 本局游戏的预测结果
            # for h in h_final:
            #     role = torch.zeros(embedding_size)
            #     x = np.zeros(embedding_size)
            #     for i in range(h.shape[0]):
            #         x[i] = h[0][i]
            #     pos = np.argmax(x)
            #     role[pos] = 1
            #     h_out.append(role)
            #
            # out.append(h_out)
        return out
