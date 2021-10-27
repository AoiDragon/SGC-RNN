import torch
import math
import numpy as np
from signed_sage_convolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep, ListModule
from utils import initialize_embedding


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
        for game in graphs:
            for mission in game:
                for vote in mission:
                    # 生成初始嵌入
                    player_num = vote["numberOfPlayers"]
                    h_0 = []  # 初始嵌入
                    for _ in range(player_num):
                        h_0.append(initialize_embedding())

                    # 进行第一层SGCN
                    h_pos, h_neg = []
                    # 第二层SGCN
                # 对当前任务进行padding（填充一个全为-1的向量），获得一个5*n的向量
            # 对当前游戏进行padding（将之前的向量cat起来），得到一个25*n的tensor
            # LSTM也需要串行计算，将当前游戏的tensor送入其中
            # 返回值是lstm最后一个有效timestep的输出，可以在for循环中设置一个保留最后结果的临时变量，边计算边更新


