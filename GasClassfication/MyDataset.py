import os
import random

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, train):
        super(MyDataset, self).__init__()
        self.train = train
        self.root_path = r"./Dataset"
        self.data = []
        self.label = []

        for file in os.listdir(self.root_path):
            with open(os.path.join(self.root_path, file), "r", encoding="utf-8") as f:
                line_list = f.readlines()
                for line in line_list:
                    line = line.strip()
                    tmp_list = line.split(" ")
                    self.label.append(int(tmp_list[0])-1)
                    tmp_data_list = []
                    for i in range(128):
                        tmp_data_list.append(float(tmp_list[i+1].split(":")[1]))
                    self.data.append(tmp_data_list)

        # Shuffle the data and labels
        temp_list = list(zip(self.data, self.label))
        random.shuffle(temp_list)
        self.data, self.label = zip(*temp_list)
        self.data = list(self.data)
        self.label = list(self.label)

        if train:
            # Used for train dataset
            del self.data[int(len(self.data) * 0.8):len(self.data)]
            del self.label[int(len(self.data) * 0.8):len(self.data)]
        else:
            # Used for test dataset
            del self.data[0:int(len(self.data) * 0.8) - 1]
            del self.label[0:int(len(self.data) * 0.8) - 1]

        self.len = len(self.data)
        self.data = torch.tensor(self.data)
        self.label = torch.tensor(self.label)

        # Normalize the data
        self.normalize()

    def normalize(self):
        self.data = F.normalize(self.data, dim=0)


    def __getitem__(self, index):
        return self.label[index], self.data[index]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    trainSet = MyDataset(train=True)
    testSet = MyDataset(train=False)
    print(len(trainSet), len(testSet))
