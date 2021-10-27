import torch
from preprocess import preprocess
from dataloader import get_loader
from param_parser import parameter_parser
from utils import generate_graph
from SGC_LSTM import SGC_LSTM

config = parameter_parser()

# preprocess(config.record_dir, config.data_dir)

data_loader = get_loader(config, 'train')

"""
for epoch in epoch_num:
    for i, data in enumerate(data_loader):
    # 第一维为完整游戏记录，第二维为玩家真实身份，注意是一个batch的数据
    records = data[0]
    labels = data[1]
        for j in range(config.batch_size):
            # record和label为记录某一局游戏信息的列表
            record = records[j]
            label = labels[j]
            # 一局游戏中所有图
            graphs = generate_graph(record)
    
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SGC_LSTM(device, config)
print(model)
model.to(device)

# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
for i, data in enumerate(data_loader):
    # 第一维为完整游戏记录，第二维为玩家真实身份，注意是一个batch的数据
    records = data[0]
    labels = data[1]

    # 这里生成一个batch的图，送入模型
    # 模型中SGCN部分的forward一个一个算
    # 算完SGCN的部分后以batch送入LSTM
    graphs = generate_graph(records)                    # 一个batch中所有游戏的图
    # graphs[0]是一场游戏的图
    # graphs[0][0]是一轮任务的图
    # graphs[0][0][0]是一轮投票的图（字典形式）
    model(graphs)



        # # record和label为记录某一局游戏信息的列表
        # record = records[j]
        # label = labels[j]
        # # 一局游戏中所有图
        # graphs = generate_graph(record)
        # print(graphs)
        # break
        # # model(graphs)












