import snntorch.spikegen
from sklearn import preprocessing
import torch


def data_preprocess(data):
    min_max_normalizer = preprocessing.MinMaxScaler(feature_range=(0, 1-1e-8))
    scaled_data = min_max_normalizer.fit_transform(data)
    return torch.tensor(scaled_data)


def rate_encoder(in_data):
    data = data_preprocess(in_data)
    spk = snntorch.spikegen.rate(data, num_steps=100)

    return spk

def latency_encoder(in_data):
    data = data_preprocess(in_data)
    spk = snntorch.spikegen.latency(data, num_steps=100)

    return spk

if __name__ == "__main__":
    from MyDataset import MyDataset
    from torch.utils.data import DataLoader

    train_dataset = MyDataset(True)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for label, data in train_dataloader:
        spk = rate_encoder(data)
        print(spk)