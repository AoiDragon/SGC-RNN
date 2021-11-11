import torch


# def predict(self, h_final, player_num):
#     """
#     根据最终嵌入输出角色预测结果
#     :param player_num: 玩家数量
#     :param h_final: 最终嵌入的列表
#     :return:
#     """
#     rank_row, rank_column, rank_tmp = [], [], []
#     role_num = self.config.embedding_size
#     #  计算玩家自己的排名向量
#     for h in h_final:
#         rank_row.append(compute_rank(h))
#
#     #  计算玩家间同一角色的排名向量
#     for i in range(role_num):
#         h_column = np.zeros(player_num)
#         for j in range(len(h_final)):
#             h_column[j] = h_final[j][i]
#         rank_tmp.append(compute_rank(torch.from_numpy(h_column)))
#     # 上述结果是列的排名，将其转化为每个角色的形式
#     for i in range(player_num):
#         h1 = []
#         for j in range(self.config.embedding_size):
#             h1.append(rank_tmp[j][i])
#         rank_column.append(h1)
#     # 确定每个玩家的角色
#     for i in range(player_num):
#         role = torch.zeros(role_num)
#         x = rank_row[i]
#         y = rank_column[i]
#         for j in range(1, role_num + 1):  # j是排名
#             index = int((x == j).nonzero())  # 找=j的x的索引
#             if y[index] <=
# x = torch.tensor([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]).to(torch.float32)
# y = torch.tensor([[0, 1], [1, 0]]).to(torch.long)
# m = torch.nn.CrossEntropyLoss()
# loss = m(x, y)
# print(loss)

# train_x = [torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]),
#            torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
#            torch.tensor([[1, 1, 1], [1, 1, 1]])]
#
# train_data1 = torch.nn.utils.rnn.pad_sequence(train_x, batch_first=True, padding_value=0)
# print(train_data1)
# y = torch.nn.utils.rnn.pack_padded_sequence(train_data1, lengths=[4, 3, 2])
#
#
#
# print(y)

x = torch.randn(3, 5)
print(x)
y = x.to(torch.float64)
print(y)
z = x.to('cuda:0')
print(z)