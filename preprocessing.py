import os
import re
import math
import torch
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# Standard ArtEmis Emotion Mapping
EMOTION_MAP = {
    'amusement': 0,
    'awe': 1,
    'contentment': 2,
    'excitement': 3,
    'anger': 4,
    'disgust': 5,
    'fear': 6,
    'sadness': 7,
    'something else': 8
}

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        text = str(text).lower()
        # Remove non-alphanumeric characters (keep spaces)
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1

        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]

class ArtemisDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Load Columns
        self.imgs = self.df["painting"].tolist()
        self.captions = self.df["utterance"].tolist()
        
        # Handle Art Styles for paths
        if "art_style" in self.df.columns:
            self.styles = self.df["art_style"].tolist()
        else:
            self.styles = [""] * len(self.df)
            
        # Handle Emotions (New Requirement)
        if "emotion" in self.df.columns:
            # Map string emotions to integers
            self.emotions = [EMOTION_MAP.get(e.lower(), 8) for e in self.df["emotion"].tolist()]
        else:
            # Fallback if column missing
            self.emotions = [8] * len(self.df)

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        style = self.styles[index]
        emotion_idx = self.emotions[index]

        # Construct Image Path
        if style:
            img_path = os.path.join(self.root_dir, style, str(img_id) + ".jpg")
        else:
            img_path = os.path.join(self.root_dir, str(img_id) + ".jpg")

        # Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        # Numericalize Caption
        numericalized_caption = [self.vocab.stoi["<start>"]]
        numericalized_caption += self.vocab.numericalize(str(caption))
        numericalized_caption += [self.vocab.stoi["<end>"]]

        # Return 3 items: Image, Caption, Emotion
        return image, torch.tensor(numericalized_caption), torch.tensor(emotion_idx)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        emotions = [item[2] for item in batch]

        imgs = torch.stack(imgs, dim=0)
        emotions = torch.tensor(emotions) # Stack integers
        
        targets = torch.nn.utils.rnn.pad_sequence(
            captions, batch_first=True, padding_value=self.pad_idx
        )

        return imgs, targets, emotions

def get_loader(root_folder, annotation_file, transform, batch_size=32, shuffle=True):
    dataset = ArtemisDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<pad>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=MyCollate(pad_idx=pad_idx),
        num_workers=0,
        pin_memory=True
    )
    return loader, dataset

# --- Embedding Helpers ---

def create_tfidf_embeddings_from_scratch(captions, vocab, embed_dim=256):
    """Computes TF-IDF embeddings manually and reduces dim via SVD."""
    print("Step 1: Computing Document Frequencies...")
    num_docs = len(captions)
    vocab_size = len(vocab)
    doc_freqs = Counter()
    
    indices = []
    values = []
    
    for doc_idx, text in enumerate(captions):
        tokens = vocab.tokenizer(text)
        word_counts = Counter(tokens)
        total_words = len(tokens)
        if total_words == 0: continue
            
        for word, count in word_counts.items():
            if word in vocab.stoi:
                word_idx = vocab.stoi[word]
                doc_freqs[word_idx] += 1
                tf = count / total_words
                indices.append([word_idx, doc_idx])
                values.append(tf)

    print("Step 2: Computing IDF...")
    idf_vec = torch.zeros(vocab_size)
    for word_idx in range(vocab_size):
        df = doc_freqs[word_idx]
        idf_vec[word_idx] = math.log(num_docs / (df + 1))

    print("Step 3: Creating Sparse TF-IDF Matrix...")
    indices_tensor = torch.LongTensor(indices).t()
    tf_values = torch.FloatTensor(values)
    
    # Apply IDF weighting
    row_indices = indices_tensor[0]
    tfidf_values = tf_values * idf_vec[row_indices]
    
    # --- FIX IS HERE: Use sparse_coo_tensor ---
    sparse_tfidf = torch.sparse_coo_tensor(
        indices_tensor, tfidf_values, torch.Size([vocab_size, num_docs])
    )

    print(f"Step 4: Dimensionality Reduction to {embed_dim}...")
    # Using PCA Lowrank (SVD approximation) for efficiency
    if vocab_size * num_docs < 100_000_000:
        dense_tfidf = sparse_tfidf.to_dense()
        U, S, V = torch.pca_lowrank(dense_tfidf, q=embed_dim)
        embeddings = torch.matmul(U, torch.diag(S))
    else:
        # Fallback for massive matrices
        from sklearn.decomposition import TruncatedSVD
        from scipy.sparse import csr_matrix
        vals = tfidf_values.numpy()
        rows = indices_tensor[0].numpy()
        cols = indices_tensor[1].numpy()
        scipy_sparse = csr_matrix((vals, (rows, cols)), shape=(vocab_size, num_docs))
        svd = TruncatedSVD(n_components=embed_dim)
        embeddings = torch.tensor(svd.fit_transform(scipy_sparse)).float()

    return embeddings

def load_pretrained_vectors(vocab, file_path, embed_dim):
    """Loads GloVe or FastText vectors from .txt/.vec file."""
    print(f"Loading embeddings from {file_path}...")
    embeddings_index = {}
    
    with open(file_path, 'r', encoding="utf-8", errors='ignore') as f:
        for i, line in enumerate(f):
            # Skip header for FastText
            if i == 0 and len(line.split()) == 2: continue
            
            values = line.split()
            word = values[0]
            if len(values) == embed_dim + 1:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
                
    vocab_size = len(vocab)
    embedding_matrix = torch.zeros((vocab_size, embed_dim))
    hits = 0
    
    for word, idx in vocab.stoi.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[idx] = torch.from_numpy(vec)
            hits += 1
        else:
            embedding_matrix[idx] = torch.randn(embed_dim)

    print(f"Mapped {hits}/{vocab_size} words.")
    return embedding_matrix