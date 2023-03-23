import snntorch.surrogate
import torch
import torch.nn as nn
import snntorch as snn


class SimpleSNN(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super(SimpleSNN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = 1000
        self.num_steps = 100
        self.spike_grad = snntorch.surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(self.num_inputs, self.num_hidden)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad, init_hidden=True)
        self.fc2 = nn.Linear(self.num_hidden, self.num_outputs)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        x = x.to(torch.float32)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)
