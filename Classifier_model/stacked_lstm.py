from __future__ import print_function
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from Dataset.data_handler.CSVFileManager import CSVFileManager
from Dataset.data_visualization.DataVisualizer import DataVisualizer


class Seq2seq(nn.Module):
    def __init__(self, num_hidden, num_cells):
        super(Seq2seq, self).__init__()
        self.num_cells = num_cells
        self.num_hidden = num_hidden
        self.cell_list = []
        if self.num_cells > 5 or self.num_hidden > 51:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        for i in range(0, num_cells):
            if i == 0:
                self.cell_list.append((nn.LSTMCell(1, num_hidden).double()).to(self.device))
            else:
                self.cell_list.append((nn.LSTMCell(num_hidden, num_hidden).double()).to(self.device))
        self.linear = nn.Linear(num_hidden, 1)

    def forward(self, iput, future=0):
        list_h = []
        list_c = []
        for i in range(0, self.num_cells):
            h_t = torch.zeros(iput.size(0), self.num_hidden, dtype=torch.double).to(self.device)
            list_h += [h_t]
            c_t = torch.zeros(iput.size(0), self.num_hidden, dtype=torch.double).to(self.device)
            list_c += [c_t]
        outputs = []
        for i, iput_t in enumerate(iput.chunk(iput.size(1), dim=1)):
            for j in range(0, self.num_cells):
                if j == 0:
                    h_t, c_t = (self.cell_list[j])(iput_t, (list_h[j], list_c[j]))
                    list_h[j] = h_t
                    list_c[j] = c_t
                else:
                    h_t, c_t = (self.cell_list[j])(list_h[j - 1], (list_h[j], list_c[j]))
                    list_h[j] = h_t
                    list_c[j] = c_t
            output = self.linear(list_h[self.num_cells - 1])
            outputs += [output]
        for i in range(future):  # if we should predict the future
            for j in range(0, self.num_cells):
                if j == 0:
                    list_h[j], list_c[j] = (self.cell_list[j])(output, (list_h[j], list_c[j]))
                else:
                    list_h[j], list_c[j] = (self.cell_list[j])(list_h[j - 1], (list_h[j], list_c[j]))
            output = self.linear(list_h[self.num_cells - 1])
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def pre_train(path):
    np.random.seed(0)
    torch.manual_seed(0)
    csv_mgr = CSVFileManager(filename=path, interval=1)
    csv_mgr.get_by_interval(interval=180)
    return csv_mgr


def train(csv_data, seq_l, num_epochs, num_hidden, num_cells, print_test=50):
    data_size = 13441
    data = csv_data.data.iloc[:data_size, 9]
    iput = data.iloc[:-1]
    target = data.iloc[1:]
    iput = torch.from_numpy(iput.values.reshape(-1, seq_length))
    target = torch.from_numpy(target.values.reshape(-1, seq_length))
    seq = Seq2seq(num_hidden=num_hidden, num_cells=num_cells)
    seq.to(seq.device)
    seq.double()
    iput = iput.to(seq.device)
    target = target.to(seq.device)
    iput.double()
    target.double()
    criteria = nn.MSELoss()
    optimizer = optim.LBFGS(seq.parameters(), lr=0.1)
    for epoch in range(num_epochs):
        print('EPOCH: ', epoch)

        def closure():
            optimizer.zero_grad()
            out = seq(iput)
            l = criteria(out, target)
            print('loss:', l.item())
            l.backward()
            return l

        optimizer.step(closure)
        if (epoch + 1) % print_test == 0:
            test(csv_data=csv_data, data_size=data_size, test_size=seq_l, seq=seq, future=1)
    return seq


def test(csv_data, data_size, test_size, seq, future):
    test_data = csv_data.data.iloc[data_size:data_size + test_size + 1, 9]
    test_visualize = pd.DataFrame(csv_data.data.iloc[data_size:data_size + test_size, 2], columns=['timestamp'])
    test_iput = test_data[:-1]
    test_target = test_data[1:]
    test_iput = torch.from_numpy(test_iput.values.reshape(-1, seq_length))
    test_target = torch.from_numpy(test_target.values.reshape(-1, seq_length))
    test_iput = test_iput.to(seq.device)
    test_target = test_target.to(seq.device)
    criteria = nn.MSELoss()
    with torch.no_grad():
        pred = seq(test_iput, future=future)
        l = criteria(pred[:, :-future], test_target)
        print('test loss:', l.item())
        y = pred.cpu().detach().numpy()
    pred = torch.squeeze(pred)
    pf = pd.DataFrame(pred[:-future].cpu().numpy(), columns=['idle'])
    pf['timestamp'] = test_visualize.timestamp.values
    ft = CSVFileManager(interval=180, df=pf)
    test_visualize.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    test_visualize['idle'] = test_data[:-1]
    ft = DataVisualizer(csv_mgr=ft, x_col='timestamp', y_col='idle')
    ft.forecast(compare_data=test_visualize, column_list=['timestamp', 'idle'])


if __name__ == '__main__':
    path = 'C://Users//Mahesh.Bhosale//PycharmProjects//Idle_bot//dataset//data//CPU_STAT//CPU_STAT_08.csv'
    csv_data_mgr = pre_train(path=path)
    seq_length = 672
    number_epochs = 1
    number_hidden = 10
    number_cells = 3
    test_size = seq_length
    seq = train(csv_data=csv_data_mgr, seq_l=seq_length, num_epochs=number_epochs, num_hidden=number_hidden,
                num_cells=number_cells)
