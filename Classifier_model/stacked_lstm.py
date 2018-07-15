from __future__ import print_function
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib
from Dataset.data_handler.CSVFileManager import CSVFileManager
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(200, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        # input = np.expand_dims(input, axis = 0)
        # print(input.size())
        # print(type(input))
        # print(type(input))
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        outputs = []

        for i, input_t in enumerate(input.chunk(chunks = input.size(1), dim=1)):
            print(input_t.size())
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    PATH = 'C://Users//Mahesh.Bhosale//PycharmProjects//Idle_bot//Dataset//Data//IO_STATS.csv'
    csv_mgr = CSVFileManager(filename=PATH, interval=60)
    # print(csv_mgr.data)
    data = csv_mgr.data.iloc[:20001, 1]
    test_data = csv_mgr.data.iloc[20002: 20803, 1]
    input = data.iloc[:-1]
    target = data.iloc[1:]
    input = input.values.reshape(200, 100)
    target = target.values.reshape(200, 100)
    test_input = test_data.iloc[:-1]
    test_target = test_data.iloc[1:]
    test_input = test_input.values.reshape(200, 4)
    test_target = test_target.values.reshape(200, 4)
    test_input = torch.from_numpy(test_input)
    test_target = torch.from_numpy(test_target)
    # print(test_input.size(), test_target.size())
    input = torch.from_numpy(input)
    target = torch.from_numpy(target)
    # print(input.size(1))
    # print(data)
    # print(input, type(input), input.shape)
    # print(target, type(target), target.shape)
    seq = Sequence()
    seq.double()
    crieterion = nn.MSELoss()
    optimizer = optim.LBFGS(seq.parameters(), lr=0.08)
    for epoch in range(15):
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
        future = 4
        pred = seq(test_input, future=future)
        loss = crieterion(pred[:, :-future], test_target)
        print(pred, test_target)
        print('test loss:', loss.item())
        y = pred.detach().numpy()
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