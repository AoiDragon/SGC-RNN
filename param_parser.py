import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()

    # Model Configuration
    parser.add_argument('--embedding_size',
                        type=int,
                        default=7,
                        help="嵌入的默认长度，默认值为10.0人局中的角色种类")

    parser.add_argument('--lstm_num',
                        type=int,
                        default=5,
                        help="lstm数量，与最大任务轮数相同，默认为5")

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
