import os
import random
import json
import torch


def collator(batch):
    return [[x[0] for x in batch], [x[1] for x in batch]]


class Avalon(torch.utils.data.Dataset):
    """Dataset class for Avalon game records"""

    def __init__(self, file_dir, mode):
        self.file_dir = file_dir
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.mode = mode
        self.preprocess()
        if mode == 'train':
            self.length = len(self.train_data)
        else:
            self.length = len(self.test_data)

    def __getitem__(self, index):
        data = self.train_data if self.mode == 'train' else self.test_data
        label = self.train_label if self.mode == 'train' else self.test_label
        return [data[index], label[index]]

    def __len__(self):
        return self.length

    def preprocess(self):
        cnt = 0
        for root, dirs, files in os.walk(self.file_dir):
            random.seed(1234)
            random.shuffle(list(files))
            for file in files:
                filename = self.file_dir + str(file)
                with open(filename, 'r', encoding="utf8") as f:
                    gameData = json.loads(f.read())

                if cnt <= 0.8 * len(files):
                    self.train_data.append(gameData)
                    self.train_label.append(gameData['rolesTensor'])
                else:
                    self.test_data.append(gameData)
                    self.test_label.append(gameData['rolesTensor'])

        print('Finished preprocessing the Avalon dataset...')


def get_loader(config, mode='train'):
    Avalon_Dataset = Avalon(config.data_dir, mode)
    data_loader = torch.utils.data.DataLoader(dataset=Avalon_Dataset,
                                              batch_size=config.batch_size,
                                              shuffle=(mode == 'train'),
                                              drop_last=True,
                                              collate_fn=collator)
    return data_loader
