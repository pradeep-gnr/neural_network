import numpy as np
import copy
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import math


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0


class NeuralNetwork():
    """
            neural network implementation using a cross entropy loss and a logistic transfer function.
    """

    def __init__(self, X, Y, hidden_layers=10, n_iterations=100, learning_rate=0.4, batch_size=50):
        self.X = X
        self.Y = Y
        self.n_classes = len(set(self.Y))
        self.H = hidden_layers
        # Weights from input layer to hidden layer.
        self.W_1 = None
        # Weights from hidden layer to the output layer.
        self.W_2 = None
        # Bias vector from input layer to hidden layer.
        self.n_iterations = n_iterations
        self.rho = learning_rate
        self.batch_size = batch_size
        self.scaler = None

    def _shuffle_input(self):
        # Super space inefficient for larger matrices.
        n_col = self.X.shape[1]
        Y = np.array(self.Y.tolist(), ndmin=2).T
        tmp = np.concatenate((self.X, Y), axis=1)
        np.random.shuffle(tmp)
        self.Y = tmp[:, n_col]
        self.X = np.delete(tmp, n_col, axis=1)

    def _standardize(self):
        self.scaler = StandardScaler()
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)

    def cross_entropy_loss(self, actual, predictions):
        tot_loss = 0
        for act, pred in zip(actual, predictions):
            loss = 0
            loss = act * math.log(pred) + (1 - act) * math.log(1 - pred)
            tot_loss += loss
        return -1 * tot_loss

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def __init_weights(self):
        self.W_1 = np.random.rand(self.X.shape[1], self.H)
        self.W_2 = np.random.rand(self.H, self.n_classes)

    def forward_pass(self, x):
        z_1 = np.dot(x, self.W_1)
        #z_1 = np.dot(x,self.W_1)
        #z_1 = z_1[0]
        # Output of hidden layer.
        a_1 = np.array([sigmoid(e) for e in z_1])
        # print "Layer 1", z_1, a_1
        z_2 = np.dot(a_1, self.W_2)
        a_2 = np.array([sigmoid(e) for e in z_2])
        #a_2 = self.softmax(a_2)
        # print "Layer 2",z_2, a_2
        return a_1, a_2

    def backward_pass(self, x_k_array, t_array, x_i_array, x_j_array):
        """
                Back propogate the gradients.
                Some terminologies.
                layer i -- output layer.
                layer j -- hidden layer.
                layer k -- input layer.
                w_ij -- weights from a node in layer i to a  node in layer j.
                x_i -- Output of a node in the softmax layer.
                x_j -- Output of a node in the hidden layer.
                x_k -- Output of a node from the input layer.
        """
        # Error gradient dE/d_w_ij -- Weights from the hidden to the output layer.
        # print self.W_2
        delta_W2 = copy.copy(self.W_2)
        delta_W1 = copy.copy(self.W_1)
        for i in range(self.n_classes):
            # dE/d_w_ij- (xi - ti) * xj
            x_i = x_i_array[i]
            t_i = t_array[i]
            for j in range(self.H):
                x_j = x_j_array[j]
                w_ij = self.W_2[j][i]
                delta = -1 * self.rho * (x_i - t_i) * x_j
                delta_W2[j][i] = delta

        # Error gradient dE/d_w_kj -- Weights from the input to the hidden
        # layer.
        for j in range(self.H):
            for k in range(len(x_k_array)):
                dE_ds = 0
                for i in range(self.n_classes):
                    x_i = x_i_array[i]
                    t_i = t_array[i]
                    x_j = x_j_array[j]
                    w_ij = self.W_2[j][i]
                    dE_ds += (x_i - t_i) * w_ij * x_j * (1 - x_j)
                x_k = x_k_array[k]
                delta = -1 * self.rho * dE_ds * x_k
                delta_W1[k][j] = delta
            # Update the Bias.

        return delta_W2, delta_W1

    def train(self):
        self.__init_weights()
        self._shuffle_input()
        self._standardize()
        for epoch in range(1, self.n_iterations + 1):
            print "Epoch : {}".format(epoch)
            batch_result = []
            for i in range(self.X.shape[0]):
                x_j_array, probas = self.forward_pass(self.X[i, :])
                actual = np.zeros(self.n_classes)
                actual[self.Y[i]] = 1
                loss = self.cross_entropy_loss(actual, probas)
                # print "Probs: {}, Loss: {}, Label: {}".format(probas, loss,
                # self.Y[i])
                result = self.backward_pass(
                    self.X[i, :], actual, probas, x_j_array)
                batch_result.append(result)
                if len(batch_result) == self.batch_size:
                    # Update Weights.
                    delta_w2s = [each[0] for each in batch_result]
                    self.W_2 += sum(delta_w2s) / len(delta_w2s)
                    delta_w1s = [each[1] for each in batch_result]
                    self.W_1 += sum(delta_w1s) / len(delta_w1s)
                    batch_result = []
            # Get network predictions.
            predictions = self.predict(self.X)
            # print predictions
            print "Epoch: {}, Error: {}".format(epoch, 1 - accuracy_score(predictions, self.Y.tolist()))

    def predict(self, X):
        probas = self.predict_proba(X)
        labels = []
        for row in probas:
            labels.append(np.argmax(row))
        return labels

    def predict_proba(self, X):
        predicted_probs = []
        for i in range(X.shape[0]):
            row = X[i, :]
            _, probas = self.forward_pass(row)
            predicted_probs.append(probas)
        return predicted_probs


if __name__ == "__main__":
    from sklearn import datasets
    #iris = datasets.load_breast_cancer()
    iris = datasets.load_iris()
    A = iris.data  # we only take the first two features.
    Z = iris.target
    nn = NeuralNetwork(A, Z)
    nn.train()
