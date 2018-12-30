from __future__ import print_function
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import math
from Dataset.data_handler.CSVFileManager import CSVFileManager
from Dataset.data_visualization.DataVisualizer import DataVisualizer


class Seq2seq(nn.Module):
    def __init__(self, num_hidden, num_cells):
        """
        Initialize the classifier
        :param num_hidden: Number of hidden units of LSTM
        :param num_cells: Number of LSTM cells in the NN, equivalent to number of layers
        """
        super(Seq2seq, self).__init__()
        self.num_cells = num_cells
        self.num_hidden = num_hidden
        self.cell_list = []
        if self.num_cells > 5 and self.num_hidden > 51:
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
        """
        Forward pass of the classifier
        :param iput: input dataframe for training/testing
        :param future: Number of future steps to be predicted
        :return: returns outputs
        """
        list_h = []
        list_c = []
        for i in range(0, self.num_cells):
            h_t = torch.zeros(iput.size(0), self.num_hidden, dtype=torch.double).to(self.device)
            list_h += [h_t]
            c_t = torch.zeros(iput.size(0), self.num_hidden, dtype=torch.double).to(self.device)
            list_c += [c_t]
        outputs = []
        print(iput.size(1))
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
    """
    Pre train work
    :param path: Complete path of the csv file for the data
    :return: Return initialized CSVFileManager object
    """
    np.random.seed(0)
    torch.manual_seed(0)
    csv_mgr = CSVFileManager(filename=path, interval=1)
    csv_mgr.get_by_interval(interval=180)
    return csv_mgr


def train(csv_data, train_to_test, data_col, time_col, seq_l, num_epochs, num_hidden, num_cells, print_test_loss=1):
    """
    train the classifier and print the training loss of the each epoch. Uses MSEloss as criteria
    :param csv_data: CSVFileManager object containing test data
    :param train_to_test: Train to test data size ratio between 0-1 exclusive
    :param data_col: # column of the target data in csv_data.data dataframe
    :param time_col: # column of the target timestamp in csv_data.data dataframe
    :param seq_l: sequence length
    :param num_epochs: Number of training cycles
    :param num_hidden: Number of hidden units
    :param num_cells: Number of LSTM cells
    :param print_test_loss: Number of epochs after which testloss is evaluated
    :return: trained LSTM classifier
    """
    total_size = csv_data.data.shape[0]
    train_size = math.floor(total_size*train_to_test)
    train_size = math.floor(train_size/seq_l)*seq_l
    data = csv_data.data.iloc[:train_size + 1, data_col]
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
            l_train = criteria(out, target)
            print('loss:', l_train.item())
            l_train.backward()
            return l_train

        optimizer.step(closure)
        if (epoch + 1) % print_test_loss == 0:
            test(csv_data=csv_data, train_size=train_size, test_size=total_size - train_size, data_col=data_col,
                 time_col=time_col, seq=seq, future=1)
    return seq


def test(csv_data, train_size, test_size, data_col, time_col, seq, future):
    """
    test the the classifier and visualizes the predicted and actual values, does not print the visualization of
    the future. Uses MSEloss as criteria
    :param csv_data: CSVFileManager object containing training data
    :param train_size: size of the train data for iloc
    :param test_size: size of the test data for iloc
    :param data_col: # column of the target data in csv_data.data dataframe
    :param time_col: # column of the target timestamp in csv_data.data dataframe
    :param seq: sequence length
    :param future: number of future steps to be predicted, can not be greater than test_size as some part of test data
    would be used for future predictions
    :return:
    """
    if future > test_size:
        raise Invalid
    test_data = csv_data.data.iloc[train_size:train_size + test_size + 1, data_col]
    test_visualize = pd.DataFrame(csv_data.data.iloc[train_size:train_size + test_size, time_col], columns=['timestamp'])
    test_iput = test_data[:-1]
    test_target = test_data[1:]
    # I am not convinced why test_size - 1, on perfect multiple train and test sizes this could break.
    test_iput = torch.from_numpy(test_iput.values.reshape(-1, test_size - 1 ))
    test_target = torch.from_numpy(test_target.values.reshape(-1, test_size - 1))
    test_iput = test_iput.to(seq.device)
    test_target = test_target.to(seq.device)
    criteria = nn.MSELoss()
    print(test_iput.size(1), test_target.size(1))
    with torch.no_grad():
        pred = seq(test_iput, future=future)
        l_test = criteria(pred[:, :-future], test_target)
        print('test loss:', l_test.item())
    pred = torch.squeeze(pred)
    pf = pd.DataFrame(pred[:-future].cpu().numpy(), columns=['idle'])
    pf['timestamp'] = test_visualize.iloc[0, :]
    ft = CSVFileManager(interval=180, df=pf)
    test_visualize.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    test_visualize['idle'] = test_data[:-1]
    ft = DataVisualizer(csv_mgr=ft, x_col='timestamp', y_col='idle')
    ft.forecast(compare_data=test_visualize, column_list=['timestamp', 'idle'])


if __name__ == '__main__':
    path = 'C://Users//Mahesh.Bhosale//PycharmProjects//Idle_bot//Dataset//data//CPU_STAT//CPU_STAT_06.csv'
    csv_data_mgr = pre_train(path=path)
    seq_length = 672
    number_epochs = 1
    number_hidden = 51
    number_cells = 3
    test_size = seq_length
    seq = train(csv_data=csv_data_mgr, seq_l=seq_length, train_to_test=0.8, data_col=9, time_col=2,
                num_epochs=number_epochs, num_hidden=number_hidden, num_cells=number_cells)
