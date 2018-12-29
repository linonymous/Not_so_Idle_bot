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
                    h_t, c_t = (self.cell_list[j])(list_h[j-1], (list_h[j], list_c[j]))
                    list_h[j] = h_t
                    list_c[j] = c_t
            output = self.linear(list_h[self.num_cells - 1])
            outputs += [output]
        for i in range(future):  # if we should predict the future
            for j in range(0, self.num_cells):
                if j == 0:
                    list_h[j], list_c[j] = (self.cell_list[j])(output, (list_h[j], list_c[j]))
                else:
                    list_h[j], list_c[j] = (self.cell_list[j])(list_h[j-1], (list_h[j], list_c[j]))
            output = self.linear(list_h[self.num_cells - 1])
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    PATH = 'C://Users//Mahesh.Bhosale//PycharmProjects//Idle_bot//dataset//data//CPU_STAT//CPU_STAT_08.csv'
    csv_mgr = CSVFileManager(filename=PATH, interval=1)
    csv_mgr.get_by_interval(interval=180)
    total_size = csv_mgr.data.shape[0]
    seq_length = 672
    data_size = 13441
    number_epochs = 1
    number_hidden = 10
    number_cells = 3
    test_size = seq_length
    data = csv_mgr.data.iloc[:data_size, 9]
    test_data = csv_mgr.data.iloc[data_size:data_size + test_size + 1, 9]
    test_visualize = pd.DataFrame(csv_mgr.data.iloc[data_size:data_size + test_size, 2], columns=['timestamp'])
    iput = data.iloc[:-1]
    target = data.iloc[1:]
    test_iput = test_data[:-1]
    test_target = test_data[1:]
    iput = torch.from_numpy(iput.values.reshape(-1, seq_length))
    target = torch.from_numpy(target.values.reshape(-1, seq_length))
    test_iput = torch.from_numpy(test_iput.values.reshape(-1, seq_length))
    test_target = torch.from_numpy(test_target.values.reshape(-1, seq_length))
    seq = Seq2seq(num_hidden=number_hidden, num_cells=number_cells)
    seq.to(seq.device)
    seq.double()
    iput = iput.to(seq.device)
    target = target.to(seq.device)
    iput.double()
    target.double()
    test_iput = test_iput.to(seq.device)
    test_target = test_target.to(seq.device)
    criteria = nn.MSELoss()
    optimizer = optim.LBFGS(seq.parameters(), lr=0.1)
    future = 1
    for epoch in range(number_epochs):
        print('EPOCH: ', epoch)
        def closure():
            optimizer.zero_grad()
            out = seq(iput)
            loss = criteria(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                pred = seq(test_iput, future=future)
                loss = criteria(pred[:, :-future], test_target)
                print('test loss:', loss.item())
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
