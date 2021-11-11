import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()

    # Model Configuration
    parser.add_argument('--embedding_size',
                        type=int,
                        default=6,
                        help="嵌入的默认长度，以标准角色配置中的数目相同")

    parser.add_argument('--layer_num',
                        type=int,
                        default=2,
                        help="sgcn层数")

    parser.add_argument('--cell_num',
                        type=int,
                        default=25,
                        help="lstm的cell数量，最多为5轮任务，每轮任务5次投票，共计25次")

    parser.add_argument('--lstm_input_size',
                        type=int,
                        default=66,
                        help="lstm的输入大小，默认为66（SGCN正嵌入32+SGCN负嵌入32+任务额外信息2）")

    parser.add_argument('--lstm_hidden_size',
                        type=int,
                        default=32,
                        help="lstm的隐藏层大小，默认为32")

    # Training Configuration
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0')

    parser.add_argument('--batch_size',
                        type=int,
                        default=32)

    parser.add_argument('--epoch_num',
                        type=int,
                        default=200)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01)

    # Directories
    parser.add_argument('--record_dir',
                        type=str,
                        default='./data/gameRecordsDataAnon.json')

    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/simplifiedGameRecord/')

    return parser.parse_args()
