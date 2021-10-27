import torch
import math
import numpy as np
from signed_sage_convolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep, ListModule
from utils import initialize_embedding, padding, judge


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
        self.positive_base_aggregator = SignedSAGEConvolutionBase(self.config.embedding_size * 4, 32).to(self.device)
        self.negative_base_aggregator = SignedSAGEConvolutionBase(self.config.embedding_size * 4, 32).to(self.device)

        self.positive_aggregators = []
        self.negative_aggregators = []
        for i in range(2):
            # 输入是32*7（6种+自己）输出暂定32*1
            self.positive_aggregators.append(SignedSAGEConvolutionDeep(32 * 7, 32).to(self.device))

            self.negative_aggregators.append(SignedSAGEConvolutionDeep(32 * 7, 32).to(self.device))

        self.positive_aggregators = ListModule(*self.positive_aggregators)
        self.negative_aggregators = ListModule(*self.negative_aggregators)

    def setup_layers(self):
        # setup_sgcn只能生成满足一轮任务的sgcn，应该调用和游戏轮数相同次，每个游戏用的不一样
        # sgcn中未包含激活函数
        self.setup_sgcn()
        for _ in range(self.cell_num):
            self.lstm.append(torch.nn.LSTMCell(self.config.lstm_input_size, self.config.lstm_hidden_size))
        self.lstm_list = ListModule(*self.lstm)

    def forward(self, graphs):
        """

        :param graphs:
        :return:
        """
        # 先串行进行SGCN计算
        # game是单独一局游戏
        for game in graphs:
            player_num = game[0][0]["numberOfPlayers"]
            game_embedding, game_embedding_list = [], []

            for _ in range(player_num):
                game_embedding_list.append([])

            for mission in game:
                add_info = []              # 存储每个玩家的任务额外信息（任务成功，是否参与任务）
                mission_embedding_list = []  # 临时存储任务中每一轮投票的嵌入,二维列表，第一维为角色

                for _ in range(player_num):
                    add_info.append(torch.ones(2))
                    mission_embedding_list.append([])

                flag = 0
                for vote in mission:
                    # 生成初始嵌入
                    h_0 = []  # 初始嵌入
                    for _ in range(player_num):
                        mission_embedding_list.append([])
                        h_0.append(initialize_embedding(self.config.embedding_size))
                    # 目前sgcn接收一轮投票的图，返回所有玩家当前的嵌入
                    # 因此这里还需要加一个玩家的维度
                    # h_pos[i]表示第i次聚合后所有点的嵌入的列表
                    h_pos, h_neg = [], []
                    # 进行第一层SGCN
                    h_pos.append(self.positive_base_aggregator(vote, "positive", h_0))
                    h_neg.append(self.negative_base_aggregator(vote, "negative", h_0))

                    # 第二层SGCN
                    for i in range(1, self.config.layer_num):
                        self.h_pos.append(
                            self.positive_aggregators[i - 1](vote, "positive", h_pos[i - 1], h_neg[i - 1]))
                        self.h_pos.append(
                            self.negative_aggregators[i - 1](vote, "negative", h_pos[i - 1], h_neg[i - 1]))
                    # h_pos[i] player_num个1*32tensor

                    for player in range(player_num):
                        mission_embedding_list[player].append(
                            torch.cat((h_pos[-1][player], h_neg[-1][player]), 1))  # player_num个1*64tensor
                        if player not in vote["nonMember"]:
                            add_info[player][1] = 0   # 任意一轮投票中未参与组队，修改add_info
                        if not flag:
                            flag = 1
                            add_info[player][0] = vote["missionResult"]  # 添加任务结果，每个角色只用只添加一次

                # 对当前任务进行padding（填充一个全为-1的向量），获得一个5*66(输出64+额外添加的2)的向量
                for player in range(player_num):
                    game_embedding_list[player] = torch.cat((game_embedding_list[player], add_info[player]), 1)  #  添加补充信息
                    game_embedding_list[player].append(padding(mission_embedding_list[player], 5, 66))  # 5*66

            # 对当前游戏进行padding（将之前的向量cat起来），得到一个25*66的tensor(并组合成一个list)
            # for player in range(player_num):
            #     game_embedding.append()

            for player in range(player_num):
                embedding = padding(game_embedding_list[player], 25, 66)  # 一个玩家在该轮任务中的嵌入，形状为25*66
                hx = torch.randn(1, self.config.lstm_hidden_size)
                cx = torch.randn(1, self.config.lstm_hidden_size)
                for i in range(self.cell_num):
                    if not judge(embedding[i]):  # 判断是不是全是-1
                        cell = self.lstm_list[i]
                        hx, cx = cell(embedding[i], (hx, cx))  # 待讨论




            # LSTM也需要串行计算，将当前游戏的tensor送入其中
            # for i in range(self.cell_num):

            # 返回值是lstm最后一个有效timestep的输出，可以在for循环中设置一个保留最后结果的临时变量，边计算边更新
