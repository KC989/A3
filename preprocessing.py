import os
import re
import pandas as pd
import torch
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# --- 1. Vocabulary Class ---
class Vocabulary:
    def __init__(self, freq_threshold=5):
        # Special tokens
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        # Convert to string to handle potential NaN values in CSV
        text = str(text).lower()
        # Remove non-alphanumeric characters (keep spaces)
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4  # Start index after special tokens

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1

        # Add words that meet the frequency threshold
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

# --- 2. Dataset Class ---
class ArtemisDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # --- CORRECTED COLUMN MAPPING ---
        # ArtEmis CSV uses 'painting' for image ID and 'utterance' for caption
        self.imgs = self.df["painting"].tolist()
        self.captions = self.df["utterance"].tolist()
        
        # We need 'art_style' to find the image in the subfolders
        if "art_style" in self.df.columns:
            self.styles = self.df["art_style"].tolist()
        else:
            # Fallback if art_style is missing (assumes flat directory)
            self.styles = [""] * len(self.df)

        # Initialize and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        style = self.styles[index]

        # Construct path: root / style / painting.jpg
        if style:
            img_path = os.path.join(self.root_dir, style, str(img_id) + ".jpg")
        else:
            img_path = os.path.join(self.root_dir, str(img_id) + ".jpg")

        # Load Image with Error Handling
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # If image is missing, return a black placeholder to prevent crashing
            # Ideally, you should clean your CSV, but this keeps the code running.
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        # Convert caption to indices
        numericalized_caption = [self.vocab.stoi["<start>"]]
        numericalized_caption += self.vocab.numericalize(str(caption))
        numericalized_caption += [self.vocab.stoi["<end>"]]

        return image, torch.tensor(numericalized_caption)

# --- 3. Collate Function (Padding) ---
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        # Stack images
        imgs = torch.stack(imgs, dim=0)

        # Pad captions to the max length in this specific batch
        targets = torch.nn.utils.rnn.pad_sequence(
            captions, batch_first=True, padding_value=self.pad_idx
        )

        return imgs, targets

# --- 4. Loader Utility ---
def get_loader(root_folder, annotation_file, transform, batch_size=32, shuffle=True):
    dataset = ArtemisDataset(root_folder, annotation_file, transform=transform)
    
    pad_idx = dataset.vocab.stoi["<pad>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=MyCollate(pad_idx=pad_idx),
        num_workers=0, # Set to 0 to avoid multiprocessing errors on some OS
        pin_memory=True
    )
    
    return loader, dataset