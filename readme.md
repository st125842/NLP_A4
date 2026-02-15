# Sentence-BERT for Natural Language Inference (NLI)

A PyTorch implementation of **Sentence-BERT (SBERT)** built from scratch. This project trains a BERT-based model on the **SNLI (Stanford Natural Language Inference)** dataset to predict the relationship between two sentences: *Entailment, Neutral, or Contradiction*.

It includes a custom implementation of the Transformer architecture, a training pipeline, and a **Dash Web Application** for real-time demonstrations.

![Web Interface](website.png)

## 📂 Project Structure

Here is an overview of the files in this repository:

| File | Description |
| :--- | :--- |
| **`bert.py`** | Contains the core model architecture classes (`BERT`, `EncoderLayer`, `MultiHeadAttention`, `Embedding`). This is the "brain" of the project. |
| **`sentenceBert.ipynb`** | The main Jupyter Notebook used for **training**, **evaluation**, and testing the SBERT model. Includes the training loop and performance metrics. |
| **`bert.ipynb`** | A notebook used for prototyping and testing individual BERT components (Attention mechanisms, Encoders) before integration. |
| **`main.py`** | The **Dash Web Application**. Run this file to launch the interactive web interface for testing custom sentences. |
| **`website.png`** | A screenshot of the web application interface. |

## 🚀 Features

* **Custom BERT Implementation**: The Transformer model is built manually in PyTorch (no Hugging Face `from_pretrained` magic!), allowing for deep understanding of the architecture.
* **Sentence Embeddings**: Converts sentences into fixed-size vectors to calculate semantic similarity.
* **NLI Classification**: Classifies sentence pairs into:
    * ✅ **Entailment**: Sentence A implies Sentence B.
    * 😐 **Neutral**: The truth of Sentence B cannot be determined from A.
    * ❌ **Contradiction**: Sentence A contradicts Sentence B.
* **Interactive Web App**: A clean UI built with Python Dash to test the model in real-time.

## 🛠️ Installation & Requirements

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the dependencies:**
    You will need PyTorch, Pandas, and Dash.
    ```bash
    pip install torch torchvision numpy pandas scikit-learn dash plotly tqdm
    ```

3.  **Dataset:**
    The project uses the SNLI dataset. The code is designed to load it automatically or via the Hugging Face `datasets` library.

## 🏃‍♂️ How to Run

### 1. Train the Model
Open `sentenceBert.ipynb` in Jupyter Notebook or VS Code to train the model.
* This will preprocess the data, train the SBERT model, and save the model weights (e.g., `sbert_model.pth`) and vocabulary (`word2id.pkl`).

### 2. Run the Web App
Once you have a trained model, run the `main.py` script to start the web interface:

```bash
python main.py