import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()

    # Model Configuration
    parser.add_argument('--embedding_size',
                        type=int,
                        default=7,
                        help="嵌入的默认长度，默认值为10.0人局中的角色种类")

    parser.add_argument('--cell_num',
                        type=int,
                        default=25,
                        help="lstm的cell数量，最多为5轮任务，每轮任务5次投票，共计25次")

    parser.add_argument('--lstm_input_size',
                        type=int,
                        default=9,
                        help="lstm的输入大小，默认为9（角色种类7+任务成功与否+本轮是否参与任务）")

    parser.add_argument('--lstm_hidden_size',
                        type=int,
                        default=32,
                        help="lstm的隐藏层大小，默认为32")

    # Training Configuration
    parser.add_argument('--batch_size',
                        type=int,
                        default=32)

    parser.add_argument('--epoch_num',
                        type=int,
                        default=200)

    # Directories
    parser.add_argument('--record_dir',
                        type=str,
                        default='./data/gameRecordsDataAnon.json')

    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/gameRecord/')

    return parser.parse_args()
