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
            h_tmp.append(self.aggregate(player, "Members", graph, feature, kind))
            h_tmp.append(self.aggregate(player, "nonMembers", graph, feature, kind))
            h_tmp.append(self.aggregate(player, "Leaders", graph, feature, kind))
            # 记得处理没找到对应节点的情况(置零)
            h_0 = feature[player]
            # 将所有类型的连接成1个向量，自己的嵌入放在最后
            for i in h_tmp:
                h_0 = torch.cat((i, h_0), 0)
            h = torch.matmul(h_0, self.weight)

            if self.bias is not None:
                h = h + self.bias
            # 这里是否需要归一化？
            # if self.norm_embed:
            #     h = F.normalize(h, p=2, dim=0)
            res.append(h)
        # res是所有玩家嵌入的列表
        return res

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

    # 对player的邻居进行聚合
    def aggregate(self, player, role, graph, feature, kind):
        # 保证了在每找到相应类型的角色时返回一个0向量
        h = torch.zeros(7, 1)
        for member in graph[role]:
            if member == player:
                continue
            elif self.judge(player, member, graph, kind):
                h += feature[member]
        h /= len(graph[role])
        return h


class SignedSAGEConvolutionDeep(SignedSAGEConvolution):
    """
    Deep Signed SAGE class for multi-layer models.
    """

    # 判断正聚合还是负聚合
    def forward(self, graph, kind, feature, base_pos, base_neg):
        """
        base_pos和base_neg分别是浅层正聚合和负聚合生成的嵌入列表
        """

        out = torch.cat((out_1, out_2, x_1), 1)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias

        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)
        return out

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

    # 对player的邻居进行聚合
    # 聚合时先确定正负，再确定种类，需要传递的参数有
    def aggregate(self, player, role, graph, feature, kind):
        # 保证了在每找到相应类型的角色时返回一个0向量 维度暂定32
        h = torch.zeros(32, 1)
        for member in graph[role]:
            if member == player:
                continue
            elif self.judge(player, member, graph, kind):
                h += feature[member]
        h /= len(graph[role])
        return h
