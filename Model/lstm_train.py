import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class Seq2Seq(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, num_layers=2, output_dim=1, device='cuda'):
        super(Seq2Seq, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = device

    def forward(self, x, hc):
        x, (h, c) = self.lstm(x, hc)
        x = x.contiguous().view(x.size()[0] * x.size()[1], self.hidden_dim)
        x = self.fc(x)
        return x, (h, c)

    def init_hidden(self, n_seqs):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, n_seqs, self.hidden_dim).zero_()).to(self.device),
                Variable(weight.new(self.num_layers, n_seqs, self.hidden_dim).zero_()).to(self.device))


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
    device = torch.device("cuda")

    # divide validation and train set
    val_idx = int(len(data) * (1 - val_frac))
    df_data = data[:val_idx]
    val = data[val_idx:]

    # Initialize the class, hidden_state, criteria and optimizer
    seq = Seq2Seq(input_dim=1, hidden_dim=256, num_layers=2, output_dim=1, device=device).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(seq.parameters(), lr=0.01)
    epoch = 81
    count = 1
    n_seqs = 5
    n_steps = 500
    x_ = None
    y_ = None
    for _ in range(epoch):
        h = seq.init_hidden(n_seqs=n_seqs)
        train_losses = []
        for x, y in get_batches(df_data, n_seqs=n_seqs, n_steps=n_steps):
            x = x.reshape((*x.shape, 1))
            x = Variable(torch.from_numpy(x).float()).to(device)
            y = y.reshape((*y.shape, 1))
            y = Variable(torch.from_numpy(y).float()).to(device)
            h = tuple([Variable(each.data).to(device) for each in h])
            x, h = seq.forward(x, hc=h)
            loss = criterion(x.view(n_seqs, n_steps, 1), y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # optimizer.zero_grad()
            # nn.utils.clip_grad_norm_(seq.parameters(), 5)
            optimizer.step()
            train_losses.append(loss.data[0])
        val_h = seq.init_hidden(n_seqs=n_seqs)
        val_losses = []
        for b, c in get_batches(val, n_seqs=n_seqs, n_steps=n_steps):
                b = b.reshape((*b.shape, 1))
                input = Variable(torch.from_numpy(b).float()).to(device)
                c = c.reshape((*c.shape, 1))
                c = Variable(torch.from_numpy(c).float()).to(device)
                output, val_h = seq.forward(input, val_h)
                val_loss = criterion(output.view(n_seqs, n_steps, 1), c)
                val_losses.append(val_loss.data[0])
        print("Epoch: {}/{}...".format(_ + 1, epoch),
              "Loss: {:.4f}...".format(np.mean(train_losses)),
              "Val Loss: {:.4f}".format(np.mean(val_losses)))
    predicted = []
    target = []
    for i in val:
        a = np.array(i)
        input = Variable(torch.from_numpy(a).float()).to(device)
        h = seq.init_hidden(n_seqs=1)
        x, (h, c) = seq.forward(input.view((1, 1, 1)), h)
        k = x.data.cpu().numpy().reshape(1).tolist()[0]
        l = input.data.cpu().numpy().tolist()
        predicted.append(k)
        target.append(l)
    df1 = pd.DataFrame(data=predicted, index=None, columns=None)
    df2 = pd.DataFrame(data=target, index=None, columns=None)
    result = pd.concat([df1, df2], axis=1)
    result.to_csv('abc.csv', encoding='utf-8', header=False, index=False)