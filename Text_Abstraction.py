import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# NLP LIBRARIES 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# SpaCy for advanced NLP
import spacy
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Downloading spaCy model...")
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU'))} GPU(s)")

#ADVANCED NLP PREPROCESSING

class AdvancedNLPPreprocessor:
    """
    Advanced NLP preprocessing using NLTK and SpaCy
    Includes: Tokenization, Lemmatization, NER, POS tagging, Dependency parsing
    """
    def __init__(self, use_spacy=True):
        self.use_spacy = use_spacy
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Keep important words for summarization
        self.keep_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 
                          'neither', 'nowhere', 'hardly', 'scarcely', 'barely'}
        self.stop_words = self.stop_words - self.keep_words
    
    def clean_text(self, text):
        """Basic text cleaning"""
        text = re.sub(r'<.*?>', '', text)  # Remove HTML
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'\[.*?\]', '', text)  # Remove citations
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        if not self.use_spacy:
            return []
        
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def get_pos_tags(self, text):
        """Get Part-of-Speech tags using NLTK"""
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        return pos_tags
    
    def lemmatize_text(self, text):
        """Lemmatization using NLTK"""
        tokens = word_tokenize(text.lower())
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized)
    
    def remove_stopwords(self, text, keep_all=False):
        """Remove stopwords while keeping important ones"""
        if keep_all:
            return text
        tokens = word_tokenize(text.lower())
        filtered = [token for token in tokens if token not in self.stop_words or token in self.keep_words]
        return ' '.join(filtered)
    
    def advanced_preprocess(self, text, extract_ner=False):
        """
        Complete NLP preprocessing pipeline
        Returns preprocessed text and metadata
        """
        # Step 1: Clean text
        text = self.clean_text(text)
        
        # Step 2: Extract entities (for context)
        entities = self.extract_entities(text) if extract_ner else []
        
        # Step 3: Lemmatization
        text = self.lemmatize_text(text)
        
        # Step 4: POS tagging info (for analysis)
        pos_tags = self.get_pos_tags(text)
        
        # Step 5: Remove stopwords
        text = self.remove_stopwords(text)
        
        metadata = {
            'entities': entities,
            'pos_distribution': self._get_pos_distribution(pos_tags),
            'token_count': len(word_tokenize(text))
        }
        
        return text, metadata
    
    def _get_pos_distribution(self, pos_tags):
        """Get distribution of POS tags"""
        pos_dist = {}
        for word, pos in pos_tags:
            pos_dist[pos] = pos_dist.get(pos, 0) + 1
        return pos_dist
    
    def get_text_statistics(self, text):
        """Extract text statistics using NLP"""
        doc = nlp(text) if self.use_spacy else None
        
        stats = {
            'sentence_count': len(sent_tokenize(text)),
            'word_count': len(word_tokenize(text)),
            'avg_word_length': np.mean([len(word) for word in word_tokenize(text)]),
            'unique_words': len(set(word_tokenize(text.lower()))),
            'lexical_diversity': len(set(word_tokenize(text.lower()))) / len(word_tokenize(text)) if len(word_tokenize(text)) > 0 else 0
        }
        
        if doc:
            stats['noun_chunks'] = len(list(doc.noun_chunks))
            stats['entities'] = len(doc.ents)
        
        return stats


class SummarizationTokenizer:
    """Custom tokenizer for text summarization"""
    def __init__(self, max_vocab_size=30000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.vocab_size = 4
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for word, _ in sorted_words[:self.max_vocab_size - 4]:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text, max_len, add_special_tokens=True):
        """Encode text to sequence"""
        words = text.split()[:max_len - 2]
        
        if add_special_tokens:
            indices = [self.word2idx['<SOS>']]
            for word in words:
                indices.append(self.word2idx.get(word, self.word2idx['<UNK>']))
            indices.append(self.word2idx['<EOS>'])
        else:
            indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # Pad sequence
        while len(indices) < max_len:
            indices.append(self.word2idx['<PAD>'])
        
        return indices[:max_len]
    
    def decode(self, indices):
        """Decode sequence to text"""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if word in ['<PAD>', '<SOS>', '<EOS>']:
                continue
            words.append(word)
        return ' '.join(words)


# CUSTOM TRANSFORMER LAYERS

class PositionalEncoding(layers.Layer):
    """Positional Encoding for Transformer"""
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.get_positional_encoding(max_len, d_model)
    
    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles
    
    def get_positional_encoding(self, max_len, d_model):
        angle_rads = self.get_angles(
            np.arange(max_len)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]


class TransformerEncoderBlock(layers.Layer):
    """Transformer Encoder with Multi-Head Attention"""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False, mask=None):
        attn_output = self.mha(x, x, x, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TransformerDecoderBlock(layers.Layer):
    """Transformer Decoder with Masked Multi-Head Attention"""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerDecoderBlock, self).__init__()
        
        self.mha1 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        self.mha2 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        # Masked self-attention
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        
        # Cross-attention with encoder output
        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        # Feed-forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3


# HYBRID ENCODER-DECODER MODEL

def build_hybrid_summarization_model(vocab_size, max_article_len=512, max_summary_len=128,
                                    embedding_dim=256, d_model=512, num_heads=8,
                                    num_encoder_layers=3, num_decoder_layers=3,
                                    dff=2048, lstm_units=256, gru_units=256, dropout_rate=0.1):
    """
    Hybrid Transformer-LSTM-GRU Seq2Seq Model for Abstractive Summarization
    
    Architecture:
    ENCODER:
    1. Embedding + Positional Encoding
    2. Transformer Encoder Blocks
    3. Bidirectional LSTM
    4. Bidirectional GRU
    
    DECODER:
    5. Embedding + Positional Encoding
    6. Transformer Decoder Blocks (with cross-attention)
    7. LSTM
    8. Dense output layer
    """
    
    # ENCODER
    encoder_inputs = layers.Input(shape=(max_article_len,), name='encoder_input')
    
    # Encoder embedding
    enc_embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    enc_embedding = layers.Dense(d_model)(enc_embedding)
    enc_embedding = PositionalEncoding(max_article_len, d_model)(enc_embedding)
    enc_embedding = layers.Dropout(dropout_rate)(enc_embedding)
    
    # Create padding mask
    enc_mask = tf.cast(tf.not_equal(encoder_inputs, 0), tf.float32)
    enc_mask = enc_mask[:, tf.newaxis, tf.newaxis, :]
    
    # Transformer encoder blocks
    enc_output = enc_embedding
    for _ in range(num_encoder_layers):
        enc_output = TransformerEncoderBlock(d_model, num_heads, dff, dropout_rate)(
            enc_output, mask=enc_mask
        )
    
    # Bidirectional LSTM
    enc_output = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)
    )(enc_output)
    
    # Bidirectional GRU
    enc_output = layers.Bidirectional(
        layers.GRU(gru_units, return_sequences=True, dropout=dropout_rate)
    )(enc_output)
    
    # ========== DECODER ==========
    decoder_inputs = layers.Input(shape=(max_summary_len,), name='decoder_input')
    
    # Decoder embedding
    dec_embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    dec_embedding = layers.Dense(d_model)(dec_embedding)
    dec_embedding = PositionalEncoding(max_summary_len, d_model)(dec_embedding)
    dec_embedding = layers.Dropout(dropout_rate)(dec_embedding)
    
    # Transformer decoder blocks
    dec_output = dec_embedding
    for _ in range(num_decoder_layers):
        dec_output = TransformerDecoderBlock(d_model, num_heads, dff, dropout_rate)(
            dec_output, enc_output
        )
    
    # LSTM for sequential processing
    dec_output = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)(dec_output)
    
    # Output layer
    outputs = layers.Dense(vocab_size, activation='softmax', name='output')(dec_output)
    
    # Create model
    model = keras.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=outputs,
        name='Hybrid_Transformer_LSTM_GRU_Summarizer'
    )
    
    return model


# ==================== DATA PREPARATION ====================

def create_sample_dataset():
    """
    Create sample dataset for text summarization
    In production, use CNN/DailyMail, XSum, or arXiv datasets
    """
    articles = [
        "Artificial intelligence has revolutionized many industries in recent years. Machine learning algorithms can now perform complex tasks that were previously thought to require human intelligence. Deep learning, a subset of machine learning, uses neural networks with multiple layers to learn from vast amounts of data. These technologies are being applied in healthcare, finance, transportation, and many other fields. The impact of AI on society continues to grow as the technology becomes more sophisticated and accessible.",
        
        "Climate change is one of the most pressing challenges facing humanity today. Rising global temperatures are causing ice caps to melt, sea levels to rise, and weather patterns to become more extreme. Scientists warn that without immediate action to reduce greenhouse gas emissions, the consequences will be severe. Renewable energy sources like solar and wind power offer promising solutions. International cooperation is essential to address this global crisis effectively.",
        
        "The human brain is an incredibly complex organ that scientists are still working to understand. It contains approximately 86 billion neurons that communicate through trillions of connections. Recent advances in neuroscience have revealed new insights into how the brain processes information, forms memories, and generates consciousness. Neuroimaging technologies allow researchers to observe brain activity in real-time. Understanding the brain better could lead to breakthroughs in treating neurological disorders.",
    ] * 100
    
    summaries = [
        "Artificial intelligence and deep learning are transforming industries through advanced machine learning algorithms and neural networks.",
        
        "Climate change threatens the planet through rising temperatures and extreme weather, requiring immediate action and renewable energy solutions.",
        
        "The human brain contains billions of neurons, and neuroscience research is revealing new insights into cognition and consciousness.",
    ] * 100
    
    return articles, summaries


# ==================== TRAINING FUNCTIONS ====================

def prepare_data(articles, summaries, nlp_preprocessor, tokenizer, 
                max_article_len=512, max_summary_len=128):
    """Prepare and preprocess data for training"""
    
    print("Preprocessing articles with NLP pipeline...")
    processed_articles = []
    processed_summaries = []
    
    for article, summary in zip(articles, summaries):
        # Advanced NLP preprocessing
        proc_article, _ = nlp_preprocessor.advanced_preprocess(article, extract_ner=True)
        proc_summary, _ = nlp_preprocessor.advanced_preprocess(summary, extract_ner=False)
        
        processed_articles.append(proc_article)
        processed_summaries.append(proc_summary)
    
    # Build vocabulary
    print("Building vocabulary...")
    all_texts = processed_articles + processed_summaries
    tokenizer.build_vocab(all_texts)
    
    # Encode texts
    print("Encoding sequences...")
    encoder_input = np.array([tokenizer.encode(text, max_article_len, add_special_tokens=False) 
                             for text in processed_articles])
    
    decoder_input = np.array([tokenizer.encode(text, max_summary_len, add_special_tokens=True) 
                             for text in processed_summaries])
    
    # Create decoder output (shifted by 1)
    decoder_output = np.zeros_like(decoder_input)
    decoder_output[:, :-1] = decoder_input[:, 1:]
    
    # Convert to one-hot for output
    decoder_output_onehot = tf.keras.utils.to_categorical(decoder_output, num_classes=tokenizer.vocab_size)
    
    return encoder_input, decoder_input, decoder_output_onehot


def plot_training_history(history):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[0].plot(history.history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('summarization_training_history.png', dpi=300)
    plt.show()
    print("✓ Saved training history")


# ==================== MAIN EXECUTION ====================

def main():
    """Main training pipeline"""
    
    print("="*80)
    print(" HYBRID TRANSFORMER-LSTM-GRU ABSTRACTIVE TEXT SUMMARIZATION")
    print(" Using Advanced NLP Preprocessing (NLTK + spaCy)")
    print("="*80)
    
    # Configuration
    MAX_VOCAB_SIZE = 30000
    MAX_ARTICLE_LEN = 512
    MAX_SUMMARY_LEN = 128
    EMBEDDING_DIM = 256
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DFF = 2048
    LSTM_UNITS = 256
    GRU_UNITS = 256
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    
    # Initialize NLP preprocessor
    print("\n1. Initializing NLP Preprocessor (NLTK + spaCy)...")
    nlp_preprocessor = AdvancedNLPPreprocessor(use_spacy=True)
    tokenizer = SummarizationTokenizer(max_vocab_size=MAX_VOCAB_SIZE)
    
    # Load dataset
    print("\n2. Loading dataset...")
    articles, summaries = create_sample_dataset()
    print(f"   Total samples: {len(articles)}")
    
    # Demonstrate NLP preprocessing
    print("\n3. Demonstrating NLP Analysis on sample article...")
    sample_stats = nlp_preprocessor.get_text_statistics(articles[0])
    print(f"   Sample Statistics: {sample_stats}")
    
    # Split data
    print("\n4. Splitting dataset...")
    articles_train, articles_temp, summaries_train, summaries_temp = train_test_split(
        articles, summaries, test_size=0.2, random_state=42
    )
    articles_val, articles_test, summaries_val, summaries_test = train_test_split(
        articles_temp, summaries_temp, test_size=0.5, random_state=42
    )
    
    print(f"   Train: {len(articles_train)} | Val: {len(articles_val)} | Test: {len(articles_test)}")
    
    # Prepare data
    print("\n5. Preparing data with NLP preprocessing...")
    enc_train, dec_train, dec_out_train = prepare_data(
        articles_train, summaries_train, nlp_preprocessor, tokenizer,
        MAX_ARTICLE_LEN, MAX_SUMMARY_LEN
    )
    
    enc_val, dec_val, dec_out_val = prepare_data(
        articles_val, summaries_val, nlp_preprocessor, tokenizer,
        MAX_ARTICLE_LEN, MAX_SUMMARY_LEN
    )
    
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Build model
    print("\n6. Building Hybrid Transformer-LSTM-GRU model...")
    model = build_hybrid_summarization_model(
        vocab_size=tokenizer.vocab_size,
        max_article_len=MAX_ARTICLE_LEN,
        max_summary_len=MAX_SUMMARY_LEN,
        embedding_dim=EMBEDDING_DIM,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dff=DFF,
        lstm_units=LSTM_UNITS,
        gru_units=GRU_UNITS,
        dropout_rate=0.1
    )
    
    print("\n" + "="*80)
    model.summary()
    print("="*80)
    
    # Compile
    print("\n7. Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_summarization_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n8. Training model...")
    print("="*80)
    
    history = model.fit(
        [enc_train, dec_train], dec_out_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([enc_val, dec_val], dec_out_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot history
    print("\n9. Plotting training history...")
    plot_training_history(history)
    
    # Save
    print("\n10. Saving model...")
    model.save('hybrid_summarization_model.h5')
    
    print("\n" + "="*80)
    print(" TRAINING COMPLETE!")
    print("="*80)
    print(" Files saved:")
    print("   - hybrid_summarization_model.h5")
    print("   - best_summarization_model.h5")
    print("   - summarization_training_history.png")
    print("="*80)
    
    return model, tokenizer, nlp_preprocessor


# ==================== INFERENCE ====================

def generate_summary(model, tokenizer, article, max_summary_len=128, max_article_len=512):
    """Generate summary for a new article"""
    # Encode input
    nlp_prep = AdvancedNLPPreprocessor()
    proc_article, _ = nlp_prep.advanced_preprocess(article)
    enc_input = np.array([tokenizer.encode(proc_article, max_article_len, add_special_tokens=False)])
    
    # Start with SOS token
    dec_input = np.array([[tokenizer.word2idx['<SOS>']]])
    
    summary_indices = [tokenizer.word2idx['<SOS>']]
    
    # Generate summary token by token
    for _ in range(max_summary_len - 1):
        predictions = model.predict([enc_input, dec_input], verbose=0)
        predicted_id = np.argmax(predictions[0, -1, :])
        
        if predicted_id == tokenizer.word2idx['<EOS>']:
            break
        
        summary_indices.append(predicted_id)
        dec_input = np.array([summary_indices])
    
    # Decode summary
    summary = tokenizer.decode(summary_indices)
    return summary


if __name__ == "__main__":
    # Train model
    model, tokenizer, nlp_preprocessor = main()
    
    # Example summarization
    print("\n" + "="*80)
    print(" EXAMPLE SUMMARIZATION")
    print("="*80)
    
    test_article = """
    Quantum computing represents a paradigm shift in computational technology. Unlike classical 
    computers that use bits as the basic unit of information, quantum computers use quantum bits 
    or qubits. These qubits can exist in multiple states simultaneously through a phenomenon called 
    superposition. Additionally, qubits can be entangled, meaning the state of one qubit can depend 
    on the state of another, regardless of the distance between them. These quantum properties allow 
    quantum computers to solve certain problems exponentially faster than classical computers. 
    Potential applications include cryptography, drug discovery, financial modeling, and optimization 
    problems. However, building practical quantum computers faces significant challenges including 
    maintaining quantum coherence and error correction.
    """
    
    print(f"\nOriginal Article:\n{test_article}\n")
    
    summary = generate_summary(model, tokenizer, test_article)
    print(f"Generated Summary:\n{summary}\n")
    
    # Show NLP analysis
    stats = nlp_preprocessor.get_text_statistics(test_article)
    entities = nlp_preprocessor.extract_entities(test_article)
    
    print("\nNLP Analysis:")
    print(f"  Statistics: {stats}")
    print(f"  Named Entities: {entities[:5]}")  # Show first 5
    
    print("\n" + "="*80)
