import torch
import numpy as np
import matplotlib.pyplot as plt
from MyDataset import MyDataset
from SNN import SimpleLeakySNN, SimpleSynapticSNN
from SpikeEncoder import rate_encoder, latency_encoder
from torch.utils.data import DataLoader


def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data)
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc * 100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc * 100:.2f}%")


def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {train_loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(spk, targets, train=True)
    print_batch_accuracy(test_spk, test_targets, train=False)
    print("\n")


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Network architecture
num_inputs = 128
num_hidden = 1024
num_outputs = 6
num_steps = 32
net = SimpleLeakySNN(num_inputs, num_hidden, num_outputs, num_steps).to(device)

# Initialize optimizer and loss function
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))
loss = torch.nn.CrossEntropyLoss()

# Create dataset and dataLoader
batch_size = 128
train_dataset = MyDataset(train=True)
test_dataset = MyDataset(train=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Training settings and record
num_epochs = 20
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []
counter = 0

for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    for targets, data in train_batch:
        # Spike encoder for data
        spk = rate_encoder(data, num_steps)

        spk = spk.to(device)
        targets = targets.to(device)
        # Forward pass
        net.train()
        spk_rec, mem_rec = net(spk)

        # Initialize the loss and sum over time
        loss_val = torch.zeros(1, dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # Gradient calculation and weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store train loss history
        train_loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_targets, test_data = next(iter(test_loader))
            test_spk = rate_encoder(test_data, num_steps)
            test_spk = test_spk.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk_rec, test_mem_rec = net(test_spk)

            # Test set loss
            test_loss = torch.zeros(1, dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem_rec[step], test_targets)
            test_loss_hist.append(test_loss.item())

            if counter % 50 == 0:
                train_printer()
            counter += 1
            iter_counter += 1

# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(train_loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
