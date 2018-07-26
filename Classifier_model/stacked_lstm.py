from __future__ import print_function
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib
from Dataset.data_handler.CSVFileManager import CSVFileManager
from Dataset.data_visualization.DataVisualizer import DataVisualizer
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.lstm3 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double).to(device)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double).to(device)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double).to(device)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double).to(device)
        h_t3 = torch.zeros((input.size(0), 51), dtype=torch.double).to(device)
        c_t3 = torch.zeros((input.size(0), 51), dtype =torch.double).to(device)
        outputs = []

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear(h_t3)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear(h_t3)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    PATH = 'C://Users//Mahesh.Bhosale//PycharmProjects//Idle_bot//Dataset//Data//IO_STAT.csv'
    csv_mgr = CSVFileManager(filename=PATH, interval=60)
    csv_mgr.get_by_interval(interval=180)
    data_visualizer = DataVisualizer(csv_mgr=csv_mgr, x_col='timestamp', y_col='tps')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    total_size = csv_mgr.data.shape[0]
    seq_length = 672
    data_size = 13441
    test_size = seq_length
    data = csv_mgr.data.iloc[:data_size, 1]
    test_data = csv_mgr.data.iloc[data_size:data_size + test_size + 1, 1]
    test_timestamp = csv_mgr.data.iloc[data_size:data_size + test_size, 0]
    input = data.iloc[:-1]
    target = data.iloc[1:]
    test_input = test_data[:-1]
    test_target = test_data[1:]
    # input = input.values.reshape(-1, seq_length)
    # target = target.values.reshape(-1, seq_length)
    # test_input = test_input.values.reshape(-1, seq_length)
    # test_target = test_target.values.reshape(-1, seq_length)
    input = torch.from_numpy(input.values.reshape(-1, seq_length))
    target = torch.from_numpy(target.values.reshape(-1, seq_length))
    test_input = torch.from_numpy(test_input.values.reshape(-1, seq_length))
    test_target = torch.from_numpy(test_target.values.reshape(-1, seq_length))
    seq = Sequence()
    # seq.to(device)
    seq.double()
    # input = input.to(device)
    # target = target.to(device)
    input.double()
    target.double()
    # test_input = test_input.to(device)
    # test_target = test_target.to(device)
    crieterion = nn.MSELoss()
    optimizer = optim.LBFGS(seq.parameters(), lr=0.06)
    future = 1
    for epoch in range(50):
        print('EPOCH: ', epoch)

        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = crieterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
    with torch.no_grad():
        future = 1
        pred = seq(test_input, future=future)
        loss = crieterion(pred[:, :-future], test_target)
        print('test loss:', loss.item())
        y = pred.detach().numpy()
    pred = torch.squeeze(pred)
    pf = pd.DataFrame(pred[:-future].numpy(), columns=['tps'])
    test_timestamp = pd.DataFrame(test_timestamp, columns=['timestamp'])
    pf['timestamp'] = test_timestamp.timestamp.values
    future = CSVFileManager(interval=180, df=pf)
    test_visualize = pd.DataFrame(test_data[:-1])
    test_visualize = pd.DataFrame(test_visualize.tps.values, index=np.arange(0, 672), columns=['tps'])
    test_visualize['timestamp'] = test_timestamp.timestamp.values
    future = DataVisualizer(csv_mgr=future, x_col='timestamp', y_col='tps')
    #     future.forecast()
    future.forecast(compare_data=test_visualize, column_list=['timestamp', 'tps'])
    # draw the result
    # plt.figure(figsize=(30, 10))
    # plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    # plt.xlabel('x', fontsize=20)
    # plt.ylabel('y', fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # def draw(yi, color):
    #     plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
    #     plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)
    # draw(y[0], 'r')
    # draw(y[1], 'g')
    # draw(y[2], 'b')
    # plt.show()
    # plt.savefig('predict%d.pdf' % epoch)
    # plt.close()