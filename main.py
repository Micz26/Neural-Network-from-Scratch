import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def one_hot_encode(data: np.ndarray) -> np.ndarray:
    y_encoded = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_encoded[rows, data] = 1
    return y_encoded

def display_loss_accuracy(loss_history: list, accuracy_history: list, filename='plot'):
    epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, epochs, 4))
    plt.title('Training Loss per Epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, epochs, 4))
    plt.title('Test Accuracy per Epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')

def normalize(X_train, X_test):
    X_max = np.max(X_train)
    X_train = X_train / X_max
    X_test = X_test / X_max
    return X_train, X_test

def xavier_init(n_in, n_out):
    low = -np.sqrt(6 / (n_in + n_out))
    high = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(low, high, (n_in, n_out))

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true)

class SingleLayerNeural:
    def __init__(self, input_size, output_size):
        self.W = xavier_init(input_size, output_size)
        self.b = xavier_init(1, output_size)

    def forward_pass(self, X):
        return sigmoid_activation(np.dot(X, self.W) + self.b)

    def backpropagate(self, X, y, learning_rate):
        error = (mse_derivative(self.forward_pass(X), y) * sigmoid_derivative(np.dot(X, self.W) + self.b))

        delta_W = (np.dot(X.T, error)) / X.shape[0]
        delta_b = np.mean(error, axis=0)

        self.W -= learning_rate * delta_W
        self.b -= learning_rate * delta_b

class DoubleLayerNeural():
    def __init__(self, input_size, output_size, hidden_size=64):
        self.W = [xavier_init(input_size, hidden_size), xavier_init(hidden_size, output_size)]
        self.b = [xavier_init(1, hidden_size), xavier_init(1, output_size)]

    def forward_pass(self, X):
        model = X
        for i in range(2):
            model = sigmoid_activation(model @ self.W[i] + self.b[i])
        return model

    def backpropagate(self, X, y, learning_rate):
        n = X.shape[0]
        biases = np.ones((1, n))
        yp = self.forward_pass(X)
        loss_grad_output = 2 * learning_rate / n * ((yp - y) * yp * (1 - yp))

        f1_output = sigmoid_activation(np.dot(X, self.W[0]) + self.b[0])
        loss_grad_hidden = np.dot(loss_grad_output, self.W[1].T) * f1_output * (1 - f1_output)

        self.W[0] -= np.dot(X.T, loss_grad_hidden)
        self.W[1] -= np.dot(f1_output.T, loss_grad_output)

        self.b[0] -= np.dot(biases, loss_grad_hidden)
        self.b[1] -= np.dot(biases, loss_grad_output)

def train_model(model, X, y, learning_rate, batch_size=100):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        model.backpropagate(X[i:i + batch_size], y[i:i + batch_size], learning_rate)

def calculate_accuracy(model, X, y):
    y_pred = np.argmax(model.forward_pass(X), axis=1)
    y_true = np.argmax(y, axis=1)
    return np.mean(y_pred == y_true)

if __name__ == '__main__':
    raw_train = pd.read_csv('fashion-mnist_train.csv')
    raw_test = pd.read_csv('fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot_encode(raw_train['label'].values)
    y_test = one_hot_encode(raw_test['label'].values)

    X_train, X_test = normalize(X_train, X_test)

    model = DoubleLayerNeural(X_train.shape[1], y_train.shape[1])
    model.backpropagate(X_train[:2], y_train[:2], 0.1)

    acc_history = []
    for _ in range(20):
        train_model(model, X_train, y_train, 0.5)
        acc_history.append(calculate_accuracy(model, X_test, y_test))
