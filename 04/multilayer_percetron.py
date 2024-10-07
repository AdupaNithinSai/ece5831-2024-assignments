import numpy as np

class MultiLayerPerceptron:
    def __init__(self, activation='sigmoid', learning_rate=0.01):
        self.net = {}
        self.activation_name = activation
        self.learning_rate = learning_rate

    def init_network(self):
        net = {}
        # Layer 1
        net['w1'] = np.random.randn(2, 3)
        net['b1'] = np.zeros(3)
        # Layer 2
        net['w2'] = np.random.randn(3, 2)
        net['b2'] = np.zeros(2)
        # Layer 3 (Output)
        net['w3'] = np.random.randn(2, 1)
        net['b3'] = np.zeros(1)
        self.net = net

    def forward(self, x):
        w1, w2, w3 = self.net['w1'], self.net['w2'], self.net['w3']
        b1, b2, b3 = self.net['b1'], self.net['b2'], self.net['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.activation(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.activation(a2)

        a3 = np.dot(z2, w3) + b3
        y = self.identity(a3)  # Output layer

        return y

    def activation(self, x):
        if self.activation_name == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_name == 'relu':
            return self.relu(x)
        elif self.activation_name == 'tanh':
            return self.tanh(x)
        elif self.activation_name == 'softmax':
            return self.softmax(x)
        else:
            raise ValueError("Unsupported activation function")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def identity(self, x):
        return x

# Test case
if __name__ == "__main__":
    mlp = MultiLayerPerceptron(activation='relu')
    mlp.init_network()
    result = mlp.forward(np.array([[1.3, 3.141592]]))
    print("Output:", result)
