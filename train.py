import numpy as np
from dataset import load_mnist
from mlp import MLP

if __name__ == "__main__":
    np.random.seed(42)
    X_train, Y_train, X_test, Y_test = load_mnist()

    net = MLP(input_size=784, hidden_size=128, output_size=10)
    net.train(X_train, Y_train, X_test, Y_test, epochs=10, batch_size=64, learning_rate=0.1)
    net.save_model("mlp_weights.npz")