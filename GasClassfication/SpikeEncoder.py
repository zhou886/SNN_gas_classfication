import snntorch.spikegen
from sklearn import preprocessing
import torch


def data_preprocess(data):
    min_max_normalizer = preprocessing.MinMaxScaler(feature_range=(0, 1 - 1e-8))
    scaled_data = min_max_normalizer.fit_transform(data)
    return torch.tensor(scaled_data, dtype=torch.float)


def rate_encoder(in_data, num_steps):
    spk = snntorch.spikegen.rate(in_data, num_steps=num_steps)

    return spk


def latency_encoder(in_data, num_steps):
    spk = snntorch.spikegen.latency(in_data, num_steps=num_steps)

    return spk


if __name__ == "__main__":
    from MyDataset import MyDataset
    from torch.utils.data import DataLoader

    train_dataset = MyDataset(True)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for label, data in train_dataloader:
        spk = rate_encoder(data, 32)
        print(spk)
