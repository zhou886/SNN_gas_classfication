from MyDataset import MyDataset
from SNN import SimpleSNN
from SpikeEncoder import encoder
import torch
import snntorch.functional as SF
from torch.utils.data import DataLoader


num_epochs = 10
num_steps = 100
num_batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

net = SimpleSNN(128, 6).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

train_dataset = MyDataset(train=True)
train_data_loader = DataLoader(train_dataset, batch_size=num_batch_size, shuffle=True)

loss_hist = []
acc_hist = []

for epoch in range(num_epochs):
    i = 0
    for label, data in train_data_loader:
        i += 1
        data = encoder(data)
        label, data = label.to(device), data.to(device)

        net.train()
        spk_rec, _ = net(data)
        loss_val = loss_fn(spk_rec, label)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        loss_hist.append(loss_val.item())

        if i % 25 == 0:
            net.eval()
            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            # check accuracy on a single batch
            acc = SF.accuracy_rate(spk_rec, label)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")