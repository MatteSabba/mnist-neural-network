import numpy as np
from dataset import load_mnist
from mlp import MLP


def render_ascii(image_784):
    """
    Riconverte un vettore di 784 pixel in una griglia 28x28 e la disegna nel terminale.
    """
    grid = image_784.reshape(28, 28)
    print("+" + "-" * 56 + "+")
    for row in grid:
        line = "".join(["██" if px > 0.7 else "▒▒" if px > 0.3 else "  " for px in row])
        print(f"|{line}|")
    print("+" + "-" * 56 + "+")


if __name__ == "__main__":
    _, _, X_test, Y_test = load_mnist()

    net = MLP()
    net.load_model("mlp_weights.npz")

    idx = np.random.randint(0, X_test.shape[0])

    # Estraggo l'immagine all'indice `idx` da X_test in modo
    # che abbia shape 2D (1, 784) e NON (784,)!
    X_single = X_test[idx : idx + 1]

    true_label = np.argmax(Y_test[idx])

    preds, probs = net.predict(X_single)

    predicted_label = preds[0]
    confidence = np.max(probs)

    print(f"\nCampione Test Set #{idx} (Input tensor shape: {X_single.shape})")
    render_ascii(X_single)
    print(f"Etichetta Reale:    {true_label}")
    print(f"Predizione Rete:    {predicted_label} (Confidenza: {confidence * 100:.2f}%)")
    print("\nDistribuzione completa Softmax sulle 10 cifre:")
    for digit, p in enumerate(probs[0]):
        bar = "█" * int(p * 30)
        print(f"  Cifra {digit}: {p * 100:6.2f}% | {bar}")