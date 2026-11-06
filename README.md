# Pure NumPy Multi-Layer Perceptron (MNIST Classifier)

Implementazione **da zero (from scratch)** di una rete neurale artificiale **Multi-Layer Perceptron (MLP)** per la classificazione di cifre scritte a mano (dataset **MNIST**), sviluppata utilizzando **esclusivamente Python e NumPy**.

---

## Scopo del Progetto

Questo repository nasce come progetto personale con uno scopo prettamente didattico e di ricerca. L'obiettivo non è competere con framework ottimizzati per la produzione (come PyTorch o TensorFlow), ma **comprendere intimamente la meccanica matematica e algoritmica** che governa il Deep Learning. 

Sviluppare un'intera architettura "from scratch" mi ha permesso di consolidare lo studio teorico applicando un approccio *first-principles*: invece di utilizzare librerie *black-box*, ho implementato e tradotto in codice puro le equazioni del calcolo differenziale (Backpropagation), l'algebra lineare (moltiplicazione tra matrici) e l'ottimizzazione stocastica (Gradient Descent).

---

## Caratteristiche Tecniche e Matematiche

* **Zero Dipendenze Esterne:** L'intero ciclo di vita del modello — dal download e parsing binario del dataset (`mnist.npz` tramite `urllib` e `np.load`) al calcolo matriciale — richiede unicamente `numpy`.
* **Inizializzazione Ibrida (He + Xavier/Glorot):**
  * **Layer 1 ($784 \to 128$):** *He Initialization* ($\mathcal{N}(0, 1) \cdot \sqrt{2 / n_{\text{in}}}$) per compensare il dimezzamento della varianza introdotto dalla funzione di attivazione **ReLU**.
  * **Layer 2 ($128 \to 10$):** *Xavier Initialization* ($\mathcal{N}(0, 1) \cdot \sqrt{1 / n_{\text{in}}}$) per mantenere i logits iniziali vicini allo zero ed evitare la saturazione precoce della **Softmax**.
* **Stabilità Numerica (Overflow & Underflow Protection):**
  * **Shifted Softmax:** Sottrazione del massimo per riga ($z_i - \max_k z_k$) prima dell'esponenziale per prevenire overflow numerici (`inf`).
  * **Clipped Cross-Entropy:** Clipping delle probabilità in $[\epsilon, 1-\epsilon]$ ($\epsilon = 10^{-15}$) per prevenire la divergenza di $\ln(0)$.
* **Backpropagation Completamente Vettorizzata:** Calcolo analitico dei gradienti tramite *Chain Rule* su mini-batch 2D senza cicli `for` sui campioni.

---

## Architettura della Rete e Mappa Dimensionale

Dato un mini-batch di input $X \in \mathbb{R}^{N \times 784}$ (immagini $28 \times 28$ normalizzate in $[0, 1]$):

| Operazione | Formula Matematica | Dimensione Tensore (`shape`) |
| :--- | :--- | :--- |
| **Input Batch** | $X$ | `(N, 784)` |
| **Hidden Pre-Activation** | $Z_1 = X \cdot W_1 + b_1$ | `(N, 784) @ (784, 128) + (1, 128) -> (N, 128)` |
| **Hidden Activation (ReLU)** | $A_1 = \max(0, Z_1)$ | `(N, 128)` |
| **Output Logits** | $Z_2 = A_1 \cdot W_2 + b_2$ | `(N, 128) @ (128, 10) + (1, 10) -> (N, 10)` |
| **Output Probabilities** | $A_2 = \text{Softmax}(Z_2)$ | `(N, 10)` |

---

## Struttura del Repository

Il codice rispetta il principio di *Separation of Concerns (SoC)* ed è suddiviso in moduli indipendenti:

```text
├── dataset.py          # Download automatico di mnist.npz, normalizzazione [0,1] e One-Hot encoding
├── mlp.py              # Classe core MLP (Forward, Backward, Loss, Update, Save/Load)
├── train.py            # Script di addestramento Mini-Batch SGD e salvataggio pesi
├── predict.py          # Script di inferenza su singola immagine con rendering ASCII da terminale
├── mlp_weights.npz     # Parametri addestrati serializzati (W1, b1, W2, b2)
└── README.md
```

---

## Installazione e Utilizzo

### 1. Requisiti
È richiesto esclusivamente `numpy`:
```bash
pip install numpy
```

### 2. Addestramento del Modello (`train.py`)
Avvia il training su $60.000$ immagini (10 epoche, `batch_size=64`) e salva automaticamente i pesi appresi in `mlp_weights.npz`(la repository contiene già il file con i pesi caricati, si può direttamente caricare tramite la funzione load_model()) :
```bash
python train.py
```
**Risultati tipici (10 epoche):**
* **Train Accuracy:** `~98.66%` (`Loss: 0.0513`)
* **Test Accuracy (su 10.000 immagini inedite):** `~97.73%` (`Loss: 0.0773`)

### 3. Inferenza e Visualizzazione (`predict.py`)
Carica istantaneamente i pesi salvati in `mlp_weights.npz`, estrae un campione casuale dal Test Set, ne disegna la griglia $28 \times 28$ in ASCII art nel terminale e mostra la distribuzione di probabilità predetta dalla Softmax:
```bash
python predict.py
```