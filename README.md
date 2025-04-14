# Yelp Sentiment Classification using LSTM and DistilBERT

This project compares two models — a custom LSTM with attention and a fine-tuned DistilBERT — for sentiment classification on Yelp reviews. The task is to classify reviews into **Negative**, **Neutral**, or **Positive** sentiment classes and evaluate model performance, interpretability, and efficiency.

---

## 📂 Project Structure

```
├── DistilBERT_Classifier.ipynb  # Fine-tuning and evaluation of DistilBERT
├── LSTM_Classifier.ipynb        # Training and evaluation of custom LSTM model
├── lime-distil.png              # LIME visualization for DistilBERT
├── TF_DB.png                    # LIME visualization for LSTM
└── README.md                    # Project overview and usage guide
```

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- NumPy, Pandas, Matplotlib
- Optuna
- LIME
- scikit-learn
- tqdm

### ✅ Installation

```bash
pip install torch transformers datasets optuna lime matplotlib scikit-learn tqdm
```

---

## 🚀 Usage

### 1. LSTM Model

- Run `LSTM_Classifier.ipynb` to:
  - Load and preprocess data
  - Train the LSTM model with attention
  - Evaluate accuracy, F1-score, and confusion matrix
  - Generate interpretability outputs using attention and LIME

### 2. DistilBERT Model

- Run `DistilBERT_Classifier.ipynb` to:
  - Load and tokenize data
  - Perform hyperparameter tuning via Optuna
  - Fine-tune DistilBERT
  - Evaluate metrics and generate interpretability visualizations

---

## ⚙️ Model Architectures

### 🧠 LSTM
- Embedding layer (dim = 100)
- Bidirectional LSTM (hidden dim = 128)
- Attention mechanism
- Dropout = 0.5
- Optimizer: Adam

### 🤖 DistilBERT
- Pretrained `distilbert-base-uncased`
- Classification head on `[CLS]` token
- Dropout: ~0.127 (tuned)
- Optimizer: AdamW (tuned)
- Batch size: 16

---

## 📊 Evaluation Results

| Model       | Test Accuracy | F1 Score | Test Loss | Training Time |
|-------------|---------------|----------|-----------|----------------|
| **LSTM**     | 96.51%        | 0.96     | 0.2184    | ~13 minutes     |
| **DistilBERT** | 80.74%     | 0.81     | 0.6395    | ~16 minutes     |

- **LSTM** showed strong generalization across input lengths and better handling of Neutral reviews.
- **DistilBERT** demonstrated higher interpretability via attention-based analysis.

---

## 🧪 Length-Based Performance

| Length Category | Range     | DistilBERT Acc | LSTM Acc |
|-----------------|-----------|----------------|----------|
| Short           | 0–50      | 82.15%         | 96.64%   |
| Medium          | 51–100    | 82.31%         | 96.38%   |
| Long            | 101–200   | 81.11%         | 96.44%   |
| Very Long       | 201+      | 76.48%         | 96.63%   |

---

## 📈 Interpretability and Visualization

- Attention weights and **LIME visualizations** were used to analyze model behavior.
- Example input: *"The food was not that bad."*

### LIME Visualizations:

#### DistilBERT  
![DistilBERT LIME](lime-distil.png)

#### LSTM  
![LSTM LIME](TF_DB.png)

- **DistilBERT** focused on `not`, `bad`, and `food` with sharp weight distribution.
- **LSTM** showed similar important tokens but with flatter attention weights.

---

## 🔬 Hyperparameter Tuning (via Optuna)

**DistilBERT Tuned Parameters**:
- Learning Rate: 4.45e-5
- Dropout: 0.127
- Optimizer: AdamW
- Weight Decay: 0.00845
- Batch Size: 16

**LSTM Best Parameters**:
- Embedding Dim: 100
- Hidden Dim: 128
- Layers: 1
- Dropout: 0.5

---

## 📌 Conclusion

- LSTM outperformed DistilBERT on generalization and Neutral review classification.
- DistilBERT offered better interpretability with sharper attention over sentiment tokens.
- Visual and length-based evaluations highlighted strengths and trade-offs for each model.

---

## 📄 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Optuna for Hyperparameter Optimization](https://optuna.org/)
- [Yelp Dataset](https://huggingface.co/datasets/Yelp)

---

## 📚 Author

Developed for the **AIGC 5500: Advanced Deep Learning** course final project.
