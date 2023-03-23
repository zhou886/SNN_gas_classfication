import snntorch.spikegen
from sklearn import preprocessing
import torch


def data_preprocess(data):
    min_max_normalizer = preprocessing.MinMaxScaler()
    scaled_data = min_max_normalizer.fit_transform(data)
    return torch.tensor(scaled_data)


def encoder(in_data):
    print(in_data)
    data = data_preprocess(in_data)
    spk = snntorch.spikegen.rate(data, num_steps=1)

    return spk


if __name__ == "__main__":
    from MyDataset import MyDataset
    from torch.utils.data import DataLoader

    device = torch.device("cpu")
    train_dataset = MyDataset(True)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    for label, data in train_dataloader:
        label, data = label.to(device), data.to(device)
        spk = encoder(data)