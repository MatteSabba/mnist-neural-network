import numpy as np


class MLP:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Inizializza i pesi (He initialization) e i bias (a zero) della rete.
        """

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size) # He Init 2.0
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size) # Xavier Init 1.0
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        """
        Applica la funzione ReLU elemento per elemento.
        """

        return np.maximum(0, Z)

    def softmax(self, Z):
        """
        Applica la funzione Softmax stabilizzata numericamente lungo le righe.
        """

        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        probabilities = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        return probabilities

    def forward(self, X):
        """
        Esegue il forward pass completo e salva i tensori intermedi per la backpropagation.
        """
        self.X = X

        # Layer 1
        self.Z1 = self.X @ self.W1 + self.b1   # Z1 = (5, 128)
        self.A1 = self.relu(self.Z1)           # A1 = (5, 128)

        # Layer 2
        self.Z2 = self.A1 @ self.W2 + self.b2  # Z2 = (5, 10)
        self.A2 = self.softmax(self.Z2)        # A2 = (5, 10)

        return self.A2

    def compute_loss(self, Y_true, A2):
        """
        Calcola la Categorical Cross-Entropy Loss media sul batch.
        """
        N = A2.shape[0]
        eps = 1e-15

        A2_clipped = np.clip(A2, eps, 1.0 - eps) # Evito di fare ln(0)

        # (-1/N * somma di Y_true * ln(A2_clipped))
        loss = -1/N * np.sum(Y_true * np.log(A2_clipped))

        return loss

    def backward(self, Y_true):
        """
        Esegue la backpropagation calcolando i gradienti dei pesi e dei bias.
        """
        N = Y_true.shape[0]

        # Layer 2
        dZ2 = (self.A2 - Y_true) / N

        self.dW2 = self.A1.T @ dZ2 # .T per la trasposta
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Layer 1
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)

        # Z1 = X * W1 + b1
        self.dW1 = self.X.T @ dZ1
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def update_params(self, learning_rate=0.1):
        """
        Aggiorna i pesi e i bias nella direzione opposta al gradiente.
        """

        self.W1 = self.W1 - learning_rate * self.dW1
        self.b1 = self.b1 - learning_rate * self.db1
        self.W2 = self.W2 - learning_rate * self.dW2
        self.b2 = self.b2 - learning_rate * self.db2

    def compute_accuracy(self, Y_true, A2):
        """
        Calcola l'accuratezza confrontando l'indice di probabilità massima con la classe vera.
        """

        y_pred = np.argmax(A2, axis=1)
        y_true_idx = np.argmax(Y_true, axis=1)

        return np.mean(y_pred == y_true_idx)

    def train(self, X_train, Y_train, X_test, Y_test, epochs=10, batch_size=64, learning_rate=0.1, decay_rate=0.02):
        """
        Addestra la rete tramite Mini-Batch Gradient Descent.
        """
        N_train = X_train.shape[0]

        for epoch in range(epochs):
            # Learning Rate Decay
            current_lr = learning_rate / (1.0 + decay_rate * epoch)

            indices = np.random.permutation(N_train)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            # Ciclo sui mini-batch
            for i in range(0, N_train, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                Y_batch = Y_shuffled[i : i + batch_size]

                self.forward(X_batch)
                self.backward(Y_batch)
                self.update_params(current_lr)


            # Valutazione di fine epoca
            train_probs = self.forward(X_train)
            train_loss = self.compute_loss(Y_train, train_probs)
            train_acc = self.compute_accuracy(Y_train, train_probs)

            test_probs = self.forward(X_test)
            test_loss = self.compute_loss(Y_test, test_probs)
            test_acc = self.compute_accuracy(Y_test, test_probs)

            print(f"Epoca {epoch + 1:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}% | "
                  f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc * 100:.2f}% | "
                  f"LR: {current_lr:.4f}")

    def predict(self, X):
        """
        Esegue l'inferenza su un input X.
        Output:
          - preds shape: (N,) -> indice della cifra predetta (0-9)
          - probs shape: (N, 10) -> distribuzione di probabilità completa
        """
        probs = self.forward(X)
        preds = np.argmax(probs, axis=1)
        return preds, probs

    def save_model(self, filepath="mlp_mnist_weights.npz"):
        """
        Salva i pesi e i bias addestrati su un file binario compresso.
        """

        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Modello salvato con successo in: {filepath}")

    def load_model(self, filepath="mlp_mnist_weights.npz"):
        """
        Carica i pesi e i bias da un file binario esistente.
        """

        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"Modello caricato con successo da: {filepath}")


