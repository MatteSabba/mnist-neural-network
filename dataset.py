import numpy as np
import os
import urllib.request


def load_mnist():
    path = "mnist.npz"
    if not os.path.exists(path):
        print("Scaricamento di mnist.npz (11 MB) in corso...")
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urllib.request.urlretrieve(url, path)

    # Caricamento nativo con NumPy
    with np.load(path) as data:
        X_train = data['x_train'].reshape(-1, 784).astype(np.float32) / 255.0
        y_train = data['y_train'].astype(int)
        X_test = data['x_test'].reshape(-1, 784).astype(np.float32) / 255.0
        y_test = data['y_test'].astype(int)

    Y_train = np.zeros((y_train.size, 10), dtype=np.float32)
    Y_train[np.arange(y_train.size), y_train] = 1.0

    Y_test = np.zeros((y_test.size, 10), dtype=np.float32)
    Y_test[np.arange(y_test.size), y_test] = 1.0

    print(f"Dataset pronto. X_train: {X_train.shape}, X_test: {X_test.shape}\n")
    return X_train, Y_train, X_test, Y_test