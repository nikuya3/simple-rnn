from csv import reader
from math import sqrt
import numpy as np


class VanillaRNN:
    def __init__(self, input_size, hidden_size, out_size):
        self.h = np.zeros(hidden_size).reshape(hidden_size, 1)
        self.w_hh = np.random.randn(hidden_size, hidden_size) * sqrt(1 / hidden_size)
        self.w_xh = np.random.randn(hidden_size, input_size) * sqrt(1 / input_size)
        self.w_yh = np.random.randn(out_size, hidden_size) * sqrt(1 / input_size)

    def step(self, x):
        self.h = np.tanh(self.w_hh.dot(self.h) + self.w_xh.dot(x))
        y = self.w_yh.dot(self.h)
        return y

data = []
with open('data/hello.csv', 'r') as file:
    hello_reader = reader(file)
    for row in hello_reader:
        new_row = []
        for column in row:
            new_row.append(int(column))
        data.append(new_row)

data = np.matrix(data)
x = data[:, :4]
y = data[:, 4]
rnn = VanillaRNN(4, 100, 4)
# forward pass
for x_i in x:
    x_i = x_i.reshape(4, 1)
    pred = rnn.step(x_i)
    print(pred)
# backward pass
