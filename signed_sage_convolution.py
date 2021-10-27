# aggregator的实现
import torch
import math
import torch.nn.functional as F


def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    std = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-std, std)


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """

    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


class SignedSAGEConvolution(torch.nn.Module):
    """
    Abstract Signed SAGE convolution class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param norm_embed: Normalize embedding -- boolean.
    :param bias: Add bias or no.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm=True,
                 norm_embed=True,
                 bias=True):
        super(SignedSAGEConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.norm_embed = norm_embed
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        size = self.weight.size()[0]
        uniform(size, self.weight)
        uniform(size, self.bias)

    # 判断当前member是正邻居还是负邻居
    def judge(self, player, member, graph, kind):
        if kind == "positive":
            if [player, member] in graph["positiveEdges"] or [member, player] in graph["positiveEdges"]:
                return True
            return False
        elif kind == "negative":
            if [player, member] in graph["negativeEdges"] or [member, player] in graph["negativeEdges"]:
                return True
            return False

    def aggregate(self, player, role, graph, feature, kind, size):
        """
        对player的邻居进行聚合
        :param player: 当前玩家
        :param role: 对什么角色的邻居聚合
        :param graph: 投票图
        :param feature: 图中节点特征（在深层aggregate时需要区分要正嵌入还是负嵌入，可以在传参时确定）
        :param kind: aggregator的类型，正->正邻居+正嵌入/负邻居+负嵌入 负->正邻居+负嵌入/负邻居+正嵌入
        :param size: 嵌入的大小
        :return:
        """
        # embedding_kind表示当前聚合的是正嵌入还是负嵌入
        # 保证了在每找到相应类型的角色时返回一个0向量
        h = torch.zeros(1, size)
        cnt = 0  # 某类角色人数
        for member in graph[role]:
            if member == player:
                continue
            elif self.judge(player, member, graph, kind):
                h += feature[member]
                cnt += 1
        h /= cnt
        return h


class SignedSAGEConvolutionBase(SignedSAGEConvolution):
    """
    Base Signed SAGE class for the first layer of the model.
    """

    # 调用的时候要指明现在是正聚合还是负聚合
    def forward(self, graph, kind, feature):
        res = []
        # kind用于区分当前进行的是正邻居聚合还是负邻居聚合, feature是存储每个节点的嵌入的列表
        for player in range(graph["numberOfPlayers"]):
            h_tmp = []
            h_tmp.append(self.aggregate(player, "Members", graph, feature, kind, 7))
            h_tmp.append(self.aggregate(player, "nonMembers", graph, feature, kind, 7))
            h_tmp.append(self.aggregate(player, "Leader", graph, feature, kind, 7))
            h_0 = feature[player]
            # 将所有类型的连接成1个向量，自己的嵌入放在最后
            for i in h_tmp:
                h_0 = torch.cat((i, h_0), 1)
            h = torch.matmul(h_0, self.weight)
            if self.bias is not None:
                h = h + self.bias
            # 这里是否需要归一化？
            if self.norm_embed:
                h = F.normalize(h, p=2, dim=0)
            res.append(h)
        # res是所有玩家嵌入的列表
        return res


class SignedSAGEConvolutionDeep(SignedSAGEConvolution):
    """
    Deep Signed SAGE class for multi-layer models.
    """

    # 判断正聚合还是负聚合
    def forward(self, graph, kind, pos_feature, neg_feature):
        """
        pos_feature和neg_feature分别是上一层中所有节点的正嵌入和负嵌入的列表
        kind表示这是positive_aggregator还是negative_aggregator
        """
        res = []
        for player in range(graph["numberOfPlayers"]):
            h_tmp = []
            # 不需要指定当前是对正嵌入聚合还是对负嵌入聚合
            # 只需要在传参时只传正嵌入或负嵌入列表供aggregate使用就可以了
            if kind == "positive":
                h_tmp.append(self.aggregate(player, "Members", graph, pos_feature, "positive", 32))  # 组队者，正邻居(positive)，正嵌入(pos_feature)
                h_tmp.append(self.aggregate(player, "nonMembers", graph, pos_feature, "positive", 32))  # 非组队者，正邻居，正嵌入
                h_tmp.append(self.aggregate(player, "Leader", graph, pos_feature, "positive", 32))  # 队长，正邻居，正嵌入
                h_tmp.append(self.aggregate(player, "Members", graph, neg_feature, "negative", 32))  # 组队者，负邻居，负嵌入
                h_tmp.append(self.aggregate(player, "nonMembers", graph, neg_feature, "negative", 32))  # 非组队者，负邻居，负嵌入
                h_tmp.append(self.aggregate(player, "Leader", graph, neg_feature, "negative", 32))  # 队长，负邻居，负嵌入
                h_0 = pos_feature[player]
            elif kind == "negative":
                h_tmp.append(self.aggregate(player, "Members", graph, neg_feature, "positive", 32))  # 组队者，正邻居，负嵌入
                h_tmp.append(self.aggregate(player, "nonMembers", graph, neg_feature, "positive", 32))  # 非组队者，正邻居，负嵌入
                h_tmp.append(self.aggregate(player, "Leader", graph, neg_feature, "positive", 32))  # 队长，正邻居，负嵌入
                h_tmp.append(self.aggregate(player, "Members", graph, pos_feature, "negative", 32))  # 组队者，负邻居，正嵌入
                h_tmp.append(self.aggregate(player, "nonMembers", graph, pos_feature, "negative", 32))  # 非组队者，负邻居，正嵌入
                h_tmp.append(self.aggregate(player, "Leader", graph, pos_feature, "negative", 32))  # 队长，负邻居，正嵌入
                h_0 = neg_feature[player]
            for i in h_tmp:
                h_0 = torch.cat((i, h_0), 1)
            h = torch.matmul(h_0, self.weight)
            if self.bias is not None:
                h = h + self.bias
            if self.norm_embed:
                h = F.normalize(h, p=2, dim=0)
            res.append(h)
        return res
