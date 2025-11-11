# Abstractive Text Summarization with Hybrid Transformer-LSTM-GRU

## Project Overview

This project implements an advanced abstractive text summarization system that combines Transformer encoder-decoder architecture with Bidirectional LSTMs and GRUs for effective sequence-to-sequence learning. The model produces fluent, human-like summaries by deeply understanding and paraphrasing the input text.

The system also features a robust NLP preprocessing pipeline using NLTK and spaCy for tokenization, lemmatization, named entity recognition, POS tagging, and stopword removal, enhancing input quality and model performance.

## Key Features

- Hybrid sequence model combining Transformer blocks, Bidirectional LSTM, and GRU layers.
- Encoder-decoder architecture with multi-head attention and positional encoding.
- Advanced NLP preprocessing with NLTK and spaCy.
- End-to-end pipeline including data preprocessing, model training, evaluation, and inference.
- Supports training on custom or publicly available datasets (e.g., CNN/DailyMail, XSum).
- Well-structured, modular, and production-ready codebase.

## Technologies Used

- TensorFlow and Keras for model implementation.
- NLTK and spaCy for natural language processing.
- Matplotlib and Seaborn for training visualization.
- Python 3.8+ recommended.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/abstractive-text-summarization.git
    cd abstractive-text-summarization
    ```

2. Install required packages:
    ```
    pip install tensorflow numpy nltk spacy matplotlib seaborn
    python -m spacy download en_core_web_sm
    ```

3. (Optional) Download and prepare datasets as needed.

## Usage

### Training

Run the training pipeline which preprocesses data, builds the hybrid model, and trains it:

```bash
python train.py
```

Outputs:
- Trained model saved as `hybrid_summarization_model.h5`
- Training history plot `summarization_training_history.png`

### Inference

Generate summaries for new text inputs using the trained model:

```python
from summarizer import generate_summary, load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer('hybrid_summarization_model.h5')
article_text = "Input your article text here..."
summary = generate_summary(model, tokenizer, article_text)
print("Summary:", summary)
```

## Dataset

Sample dataset with articles and corresponding summaries is included for demonstration purposes. For production, use larger datasets such as:

- CNN/DailyMail
- XSum
- arXiv scientific papers

Datasets can be loaded using Hugging Face Datasets or manually prepared CSV files.

## Model Architecture

- **Encoder**: Embedding → Positional Encoding → Transformer Encoder Layers → Bi-LSTM → Bi-GRU
- **Decoder**: Embedding → Positional Encoding → Transformer Decoder Layers → LSTM → Output Dense Layer with Softmax
- Attention mechanisms enable the model to focus on relevant input segments for each output token.

## Evaluation

The project supports tracking of training and validation accuracy and loss, with visualization scripts to analyze performance trends.

## Contributions

Contributions for dataset integration, performance optimization, and deployment are welcome.
