from csv import reader
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np


def calculate_accuracy(scores, y):
    predicted_classes = np.argmax(scores, axis=1)
    correct_classes = len(np.where(predicted_classes == y)[0])
    return correct_classes / len(predicted_classes)


def probs(scores):
    """
    Calculates the probabilities out of a neural networks class scores.
    :param scores: The score matrix of form (N x K), where N is the number of observations and K is the number of classes.
    :return: The probabilities of the same form as the input scores.
    """
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def calculate_cross_entropy_loss(s, y):
    """
    Calculates the loss of a score matrix depending on the ground truth labels.
    This method uses cross entropy loss (from Softmax).
    :param s: The score matrix of form (N x K), where N is the number of observations and K is the number of classes.
    :param y: The ground truth label vector of length N.
    :param w: The weight matrix of the output layer of form (H x K), where H is the size of the previous layer.
    :param lambda_: The regularization loss hyperparameter.
    :return: The cross-entropy loss, where 0 indicates a perfect match between s and y
    and +Inf indicates a perfect mismatch.
    """
    probabilities = probs(s)
    log_probabilities = - np.log(probabilities[range(len(y)), y])
    data_loss = np.sum(log_probabilities) / len(y)
    return data_loss


def cross_entropy_loss_gradient(s, y):
    """
    Calculates the gradient of the hinge loss function by the scores.
    The gradient formula is { ds_j / dL = e^s_j / sum e^j, ds_y_i / dL = e^s_y_i / sum e^j - 1 }.
    :param s: The score parameter of the loss function.
    :param y: The ground truth label parameter of the loss function.
    :return: The gradient as a matrix of the same shape as `s`.
    """
    dscores = probs(s)
    dscores[range(len(y)), y] -= 1
    dscores /= len(y)
    return dscores


def update_parameter(x, dx, epoch, learning_rate, m, v, beta1, beta2, eps):
    """
    Updates the parameter x using its gradient dx.
    :param x: The parameter to be updated.
    :param dx: The gradient of the parameter to be updated.
    :param epoch: The current training epoch.
    :param learning_rate: The learning rate of the network. Indicates the size of learning steps.
    :param m: The current momentum.
    :param v: The current velocity.
    :param beta1: Hyperparameter for the Adam parameter update. Recommended to be .9.
    :param beta2: Hyperparameter for the Adam parameter update. Recommended to be .999.
    :param eps: Hyperparameter for the Adam parameter update. Recommended to be 1e-8.
    :return: The updated parameter of the same type as x.
    """
    m = beta1 * m + (1 - beta1) * dx
    mt = m / (1 - beta1 ** epoch)
    v = beta2 * v + (1 - beta2) * (np.square(dx))
    vt = v / (1 - beta2 ** epoch)
    update = - learning_rate * mt / (np.sqrt(vt) + eps)
    x += update
    return x, m, v


class VanillaRNN:
    def __init__(self, input_size, hidden_size, out_size):
        self.h = [np.zeros(hidden_size).reshape(hidden_size, 1)]
        self.x = []
        self.w_hh = np.random.randn(hidden_size, hidden_size) * sqrt(1 / hidden_size)
        self.w_xh = np.random.randn(hidden_size, input_size) * sqrt(1 / input_size)
        self.w_yh = np.random.randn(out_size, hidden_size) * sqrt(1 / input_size)
        self.dh = np.ones(self.h[0].shape)
        self.dw_hh = np.ones(self.w_hh.shape)
        self.dw_xh = np.ones(self.w_xh.shape)
        self.dw_yh = np.ones(self.w_yh.shape)
        self.m_whh = .0
        self.m_wxh = .0
        self.m_wyh = .0
        self.v_whh = .0
        self.v_wxh = .0
        self.v_wyh = .0

    def predict(self, x):
        y = self.step(x)
        return np.argmax(y)

    def step(self, x):
        # f = self.w_hh.dot(self.h[-1]) + self.w_xh.dot(x)
        # f[f < 0] = 0
        # self.h.append(f)
        self.h.append(np.tanh(self.w_hh.dot(self.h[-1]) + self.w_xh.dot(x)))
        self.x.append(x)
        y = self.w_yh.dot(self.h[-1])
        return y

    def back(self, dy):
        # Backpropagation through time
        self.dw_yh += dy.dot(self.h[-1].T)
        self.dh = self.dw_yh.T.dot(dy)
        self.dw_xh += self.dh.dot(self.x[-1].T)# * (1 - self.h[-1] ** 2)
        self.dw_hh += self.dh.dot(self.h[-2].T)# * (1 - self.h[-1] ** 2)
        # f1 = self.dh.dot(self.x[-1].T)
        # f1[f1 < 0] = 0
        # self.dw_xh += f1
        # f2 = self.dh.dot(self.h[-2].T)
        # f2[f2 < 0] = 0
        # self.dw_hh += f2
        del self.h[-1]
        del self.x[-1]

    def update(self, eta, epoch, n):
        self.w_hh, self.m_whh, self.v_whh =\
            update_parameter(self.w_hh, self.dw_hh, epoch, eta, self.m_whh, self.v_whh, beta1, beta2, eps)
        self.w_xh, self.m_wxh, self.v_wxh =\
            update_parameter(self.w_xh, self.dw_xh, epoch, eta, self.m_wxh, self.v_wxh, beta1, beta2, eps)
        self.w_yh, self.m_wyh, self.v_wyh =\
            update_parameter(self.w_yh, self.dw_yh, epoch, eta, self.m_wyh, self.v_wyh, beta1, beta2, eps)
        # self.w_hh -= eta * self.dw_hh
        # self.w_xh -= eta * self.dw_xh
        # self.w_yh -= eta * self.dw_yh


data = []
with open('data/hello.csv', 'r') as file:
    hello_reader = reader(file)
    for row in hello_reader:
        new_row = []
        for column in row:
            new_row.append(int(column))
        data.append(new_row)

beta1 = .9  # Hyperparameter for Adam parameter update.
beta2 = .999  # Hyperparameter for Adam parameter update.
eps = 1e-8  # Hyperparameter for Adam parameter update.
epochs = 100
eta = 1e-4

data = np.array(data)
unique_chars = data.shape[1] - 1
x = data[:, :unique_chars]
y = data[:, unique_chars]
n = len(x)  # number of training observations
k = np.unique(y).shape[0]  # number of classes
rnn = VanillaRNN(unique_chars, 200, k)
ch2n = {'h': x[0].reshape(x.shape[1], 1), 'e': x[1].reshape(x.shape[1], 1), 'l': x[2].reshape(x.shape[1], 1)}
n2ch = {y[0]: 'e', y[1]: 'l', y[3]: 'o'}
# ch2n = {
#     'h': x[0].reshape(x.shape[1], 1),
#     'e': x[1].reshape(x.shape[1], 1),
#     'l': x[2].reshape(x.shape[1], 1),
#     'o': x[3].reshape(x.shape[1], 1),
#     ',': x[4].reshape(x.shape[1], 1),
#     ' ': x[5].reshape(x.shape[1], 1),
#     'w': x[6].reshape(x.shape[1], 1),
#     'r': x[8].reshape(x.shape[1], 1),
#     'd': x[10].reshape(x.shape[1], 1),
# }
# n2ch = {
#     y[0]: 'e',
#     y[1]: 'l',
#     y[3]: 'o',
#     y[4]: ',',
#     y[5]: ' ',
#     y[6]: 'w',
#     y[8]: 'r',
#     y[10]: 'd',
#     y[11]: '!'
# }
losses = []
accs = []
for epoch in range(1, epochs + 1):
    scores = np.empty((n, k))
    pred_word = ''
    for i in range(len(x)):
        x_i = x[i].reshape(x.shape[1], 1)
        y_i = y[i]
        pred = rnn.step(x_i)
        pred_word += n2ch[np.argmax(pred)]
        scores[i] = pred.reshape(k)
    loss = calculate_cross_entropy_loss(scores, y)
    acc = calculate_accuracy(scores, y)
    print(pred_word)
    print(epoch, loss, acc)
    losses.append(loss)
    accs.append(acc)
    dscores = cross_entropy_loss_gradient(scores, y)
    for i in range(len(dscores) - 1, 0, -1):
        rnn.back(dscores[i].reshape(dscores[i].shape[0], 1))
    rnn.update(eta, epoch, n)

# test
word = 'hello'
letter = word[0]
print(letter, end='')
for nr in range(len(word)):
    pred = rnn.predict(ch2n[letter])
    letter = n2ch[pred]
    print(letter, end='')
    if letter == word[-1]:
        break

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(accs)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
