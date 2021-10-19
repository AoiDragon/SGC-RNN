import torch
import math
import numpy as np
from signed_sage_convolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep, ListModule


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

        self.lstm_num = self.config.lstm_num
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
        self.setup_sgcn()
        # sgcn中未包含激活函数

    def forward(self, graphs):
        """

        :param graphs:
        :return:
        """
        mission_num = len(graphs)
        for mission in graphs:
            vote_num = len(mission)
