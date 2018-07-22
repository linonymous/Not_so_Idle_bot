import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class Seq2Seq(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, num_layers=2, output_dim=1):
        super(Seq2Seq, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hc):
        x, (h, c) = self.lstm(x, hc)
        x = x.contiguous().view(x.size()[0] * x.size()[1], self.hidden_dim)
        x = self.fc(x)
        return x, (h, c)

    def init_hidden(self, n_seqs):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, n_seqs, self.hidden_dim).zero_()),
                Variable(weight.new(self.num_layers, n_seqs, self.hidden_dim).zero_()))


def get_batches(arr, n_steps, n_seqs):
    batch_size = n_seqs * n_steps
    num_batches = len(arr)//batch_size
    arr = arr[:num_batches * batch_size]
    arr = arr.reshape((n_seqs, -1))
    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:, n:n + n_steps]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    path = "C:\\Users\\Swapnil.Walke\\Idle_bot\\Dataset\\Data\\CPU_STAT_FINAL.csv"
    df = pd.read_csv(path)
    data = 100 - df["%idle"]
    data = data.as_matrix()
    val_frac = 0.1

    # divide validation and train set
    val_idx = int(len(data) * (1 - val_frac))
    df_data = data[:val_idx]
    val = data[val_idx:]

    # Initialize the class, hidden_state, criteria and optimizer
    seq = Seq2Seq(input_dim=1, hidden_dim=256, num_layers=2, output_dim=1)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(seq.parameters(), lr=0.0001)
    epoch = 10
    count = 1
    n_seqs = 5
    n_steps = 100
    a=0
    for _ in range(epoch):
        h = seq.init_hidden(n_seqs=n_seqs)
        for x, y in get_batches(df_data, n_seqs=n_seqs, n_steps=n_steps):
            x = x.reshape((*x.shape, 1))
            x = Variable(torch.from_numpy(x).float())
            y = y.reshape((*y.shape, 1))
            y = Variable(torch.from_numpy(y).float())
            h = tuple([Variable(each.data) for each in h])
            x, h = seq.forward(x, hc=h)
            loss = criterion(x.view(n_seqs, n_steps, 1), y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # optimizer.zero_grad()
            # nn.utils.clip_grad_norm_(seq.parameters(), 5)
            optimizer.step()
            if _ % count == 0:
                val_h = seq.init_hidden(n_seqs=n_seqs)
                val_losses = []
                for b, c in get_batches(val, n_seqs=n_seqs, n_steps=n_steps):
                    b = b.reshape((*b.shape, 1))
                    input = Variable(torch.from_numpy(b).float())
                    c = c.reshape((*c.shape, 1))
                    c = Variable(torch.from_numpy(c).float())
                    output, val_h = seq.forward(input, val_h)
                    val_loss = criterion(output.view(n_seqs, n_steps, 1), c)
                    val_losses.append(val_loss.data[0])
                print("Epoch: {}/{}...".format(_ + 1, epoch),
                      "Step: {}...".format(count),
                      "Loss: {:.4f}...".format(loss.data[0]),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))