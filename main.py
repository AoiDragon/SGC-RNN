from param_parser import parameter_parser
from train import Train
from preprocess import preprocess
from dataloader import get_loader


config = parameter_parser()

# preprocess(config.record_dir, config.data_dir)

if __name__ == '__main__':
    data_loader = get_loader(config, 'train')
    Train(config, data_loader)













