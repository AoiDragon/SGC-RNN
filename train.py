import torch
from dataloader import get_loader
from utils import generate_graph
from SGC_LSTM import SGC_LSTM
import time


def Train(config, data_loader):
    # data_loader = get_loader(config, 'train')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(device)
    model = SGC_LSTM(device, config)
    print(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.epoch_num):
        start = time.time()
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        for i, data in enumerate(data_loader):
            # 第一维为完整游戏记录，第二维为玩家真实身份，注意是一个batch的数据
            optimizer.zero_grad()
            records = data[0]
            labels = data[1]
            graphs = generate_graph(records)  # 一个batch中所有游戏的图
            out = model(graphs)
            LOSS = torch.nn.CrossEntropyLoss()
            loss = 0
            for j in range(config.batch_size):  # 每一局游戏
                loss += LOSS(out[j], labels[j].to(device))
            loss /= config.batch_size
            end = time.time()
            print("epoch %d , loss = %f, time = %f" % (epoch, loss, end - start))
            start = end
            loss.backward()
            optimizer.step()
