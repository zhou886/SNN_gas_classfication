import os
import torch
from torch.utils.data import Dataset


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
                    self.label.append(int(tmp_list[0]))
                    tmp_data_list = []
                    for i in range(128):
                        tmp_data_list.append(float(tmp_list[i+1].split(":")[1]))
                    self.data.append(tmp_data_list)

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

    def __getitem__(self, index):
        return self.label[index], self.data[index]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    trainSet = MyDataset(train=True)
    testSet = MyDataset(train=False)
    print(len(trainSet), len(testSet))
    for data in trainSet:
        print(data)
