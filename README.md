# 🤖 Humor Classification with Fine-Tuned Transformer

This project fine-tunes a pre-trained transformer-based encoder from Hugging Face to classify humorous content. By freezing the encoder and training only a custom classification head, the model achieves significantly improved accuracy on a labeled dataset. Through targeted error analysis and model refinement, the F1-score increased from **48%** to **75%**—making it suitable for real-time humor detection, such as chatbot integration on entertainment platforms.

---

## 📌 Features

- Fine-tuning of a Hugging Face transformer for binary humor classification.
- Only the classification head is trained; the base encoder remains frozen.
- Tokenization and preprocessing tailored for short, informal texts (e.g., jokes, tweets).
- Custom error analysis pipeline for misclassified samples.
- End-to-end reproducible training pipeline.
- Significant performance improvement on validation/test sets.

---

## 🧠 Model Overview

- **Base Model:** [Hugging Face Transformers](https://huggingface.co/models) (e.g., `bert-base-uncased`, `distilbert-base-uncased`)
- **Task:** Humor Detection (Binary Classification)
- **Architecture:** Frozen encoder + Trainable classification head (dense layers)

---

## 🔧 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/humor-classification-transformer.git
cd humor-classification-transformer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:
- `transformers`
- `datasets`
- `scikit-learn`
- `pandas`
- `numpy`
- `torch`

---

## 📁 Project Structure

```
├── data/                     # Raw and processed data
├── models/                   # Saved model checkpoints
├── outputs/                  # Misclassified samples and logs
├── src/
│   ├── preprocess.py         # Tokenization and preprocessing
│   ├── train.py              # Model training pipeline
│   ├── evaluate.py           # Evaluation and metrics
│   ├── error_analysis.py     # Misclassification analysis
├── requirements.txt
└── README.md
```

---

## 📊 Performance

| Metric     | Before (Baseline) | After Fine-Tuning |
|------------|-------------------|-------------------|
| F1-Score   | 0.48              | 0.75              |
| Accuracy   | ~50%              | ~76%              |
| Precision  | ↑ Improved        | ↑                 |
| Recall     | ↑ Improved        | ↑                 |

---

## 🔍 Error Analysis

A comprehensive review of **75+** misclassified samples revealed:

- Ambiguity in sarcasm vs. humor
- Edge cases with cultural or idiomatic references
- Short or contextless inputs

Insights from this analysis informed preprocessing tweaks and classification thresholds.

---

## 🚀 Use Case: Chatbot Integration

This model is optimized for low-latency inference, making it ideal for real-time chatbot systems, humor moderation tools, or content tagging in entertainment applications.

---

## 📦 How to Run

### Train the Model

```bash
python src/train.py --model bert-base-uncased --epochs 3 --freeze_encoder
```

### Evaluate the Model

```bash
python src/evaluate.py --checkpoint models/best_model.pt
```

### Run Error Analysis

```bash
python src/error_analysis.py --input outputs/predictions.csv
```

---

## 🧪 Future Improvements

- Add multilingual support for humor detection across languages.
- Integrate prompt-based few-shot learning for rare humor types.
- Experiment with adapters and LoRA for lighter deployment.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📜 License

MIT License. See `LICENSE` file for details.
