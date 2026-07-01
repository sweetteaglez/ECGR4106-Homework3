# =========================
# ECGR 4106 Homework 3
# Name: Samantha Gonzalez
# Sequence-to-Sequence Machine Translation
# =========================

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import re
import os

# BLEU score
import nltk
nltk.download("punkt")
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

os.makedirs("results", exist_ok=True)


# =========================
# Load Dataset
# =========================

filename = "vast_english_french.txt"

# The file should be uploaded to Colab before running this cell
with open(filename, "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Total lines:", len(lines))
print(lines[:3])

# =========================
# Text Cleaning
# =========================

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-ZÀ-ÿ?.!,']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

pairs = []

for line in lines:
    parts = line.strip().split("\t")

    # Most translation files are English \t French
    if len(parts) >= 2:
        eng = clean_text(parts[0])
        fra = clean_text(parts[1])

        if len(eng) > 0 and len(fra) > 0:
            pairs.append((eng, fra))

print("Number of sentence pairs:", len(pairs))
print(pairs[:5])

# =========================
# Vocabulary Class
# =========================

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

class Vocabulary:
    def __init__(self):
        self.word2idx = {
            PAD_TOKEN: 0,
            SOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3
        }

        self.idx2word = {
            0: PAD_TOKEN,
            1: SOS_TOKEN,
            2: EOS_TOKEN,
            3: UNK_TOKEN
        }

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            index = len(self.word2idx)
            self.word2idx[word] = index
            self.idx2word[index] = word

    def sentence_to_indices(self, sentence, max_len):
        words = sentence.split()
        indices = [self.word2idx.get(word, self.word2idx[UNK_TOKEN]) for word in words]
        indices = [self.word2idx[SOS_TOKEN]] + indices + [self.word2idx[EOS_TOKEN]]

        if len(indices) < max_len:
            indices += [self.word2idx[PAD_TOKEN]] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
            indices[-1] = self.word2idx[EOS_TOKEN]

        return indices

    def indices_to_sentence(self, indices):
        words = []

        for idx in indices:
            word = self.idx2word.get(int(idx), UNK_TOKEN)

            if word == EOS_TOKEN:
                break

            if word not in [SOS_TOKEN, PAD_TOKEN]:
                words.append(word)

        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

# =========================
# Keep Same 80/20 Split for All Problems
# =========================

train_pairs, val_pairs = train_test_split(
    pairs,
    test_size=0.2,
    random_state=SEED
)

print("Train pairs:", len(train_pairs))
print("Validation pairs:", len(val_pairs))

# =========================
# Build English and French Vocabularies
# =========================

eng_vocab = Vocabulary()
fra_vocab = Vocabulary()

for eng, fra in train_pairs:
    eng_vocab.add_sentence(eng)
    fra_vocab.add_sentence(fra)

print("English vocab size:", len(eng_vocab))
print("French vocab size:", len(fra_vocab))

# =========================
# Translation Dataset
# =========================

MAX_LEN = 15

class TranslationDataset(Dataset):
    def __init__(self, sentence_pairs, source_vocab, target_vocab, max_len=MAX_LEN):
        self.sentence_pairs = sentence_pairs
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, index):
        source_sentence, target_sentence = self.sentence_pairs[index]

        source_indices = self.source_vocab.sentence_to_indices(source_sentence, self.max_len)
        target_indices = self.target_vocab.sentence_to_indices(target_sentence, self.max_len)

        return (
            torch.tensor(source_indices, dtype=torch.long),
            torch.tensor(target_indices, dtype=torch.long)
        )


def make_loaders(train_pairs, val_pairs, source_vocab, target_vocab, batch_size=64):
    train_dataset = TranslationDataset(train_pairs, source_vocab, target_vocab)
    val_dataset = TranslationDataset(val_pairs, source_vocab, target_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# =========================
# Baseline Encoder and Decoder
# =========================

class EncoderGRU(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super(EncoderGRU, self).__init__()

        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, source):
        embedded = self.embedding(source)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class DecoderGRU(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super(DecoderGRU, self).__init__()

        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_token, hidden):
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)

        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc(output.squeeze(1))

        return prediction, hidden


class Seq2SeqBaseline(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqBaseline, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(source)

        input_token = target[:, 0]

        for t in range(1, target_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t, :] = output

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = target[:, t] if use_teacher_forcing else top1

        return outputs


# =========================
# Training and Validation Functions
# =========================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for source, target in loader:
        source = source.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(source, target)

        output_dim = output.shape[-1]

        loss = criterion(
            output[:, 1:].reshape(-1, output_dim),
            target[:, 1:].reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for source, target in loader:
            source = source.to(device)
            target = target.to(device)

            output = model(source, target, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]

            loss = criterion(
                output[:, 1:].reshape(-1, output_dim),
                target[:, 1:].reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(loader)

# =========================
# Prediction and Metrics
# =========================

def translate_sentence(model, sentence, source_vocab, target_vocab, max_len=MAX_LEN):
    model.eval()

    source_indices = source_vocab.sentence_to_indices(sentence, max_len)
    source_tensor = torch.tensor(source_indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(source_tensor)

        input_token = torch.tensor([target_vocab.word2idx[SOS_TOKEN]], dtype=torch.long).to(device)

        outputs = []
        attention_list = []

        for _ in range(max_len):
            if isinstance(model.decoder, AttentionDecoderGRU):
                output, hidden, attention_weights = model.decoder(input_token, hidden, encoder_outputs)
                attention_list.append(attention_weights.cpu().numpy()[0])
            else:
                output, hidden = model.decoder(input_token, hidden)

            predicted_id = output.argmax(1).item()

            if predicted_id == target_vocab.word2idx[EOS_TOKEN]:
                break

            outputs.append(predicted_id)
            input_token = torch.tensor([predicted_id], dtype=torch.long).to(device)

    translated_sentence = target_vocab.indices_to_sentence(outputs)

    return translated_sentence, attention_list


def calculate_bleu(reference, prediction):
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()

    if len(prediction_tokens) == 0:
        return 0.0

    smoothing = SmoothingFunction().method1

    return sentence_bleu(
        [reference_tokens],
        prediction_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing
    )


def evaluate_model(model, pairs, source_vocab, target_vocab, sample_count=5):
    exact_matches = 0
    bleu_scores = []
    sample_outputs = []

    for source_sentence, target_sentence in pairs:
        prediction, attention = translate_sentence(
            model,
            source_sentence,
            source_vocab,
            target_vocab
        )

        exact_match = prediction.strip() == target_sentence.strip()

        if exact_match:
            exact_matches += 1

        bleu = calculate_bleu(target_sentence, prediction)
        bleu_scores.append(bleu)

    sequence_accuracy = 100 * exact_matches / len(pairs)
    average_bleu = np.mean(bleu_scores)

    for i in range(sample_count):
        source_sentence, target_sentence = pairs[i]
        prediction, attention = translate_sentence(model, source_sentence, source_vocab, target_vocab)
        bleu = calculate_bleu(target_sentence, prediction)
        exact_match = prediction.strip() == target_sentence.strip()

        sample_outputs.append({
            "source": source_sentence,
            "target": target_sentence,
            "prediction": prediction,
            "exact_match": exact_match,
            "bleu_score": bleu
        })

    return sequence_accuracy, average_bleu, sample_outputs

# =========================
# Helper Functions
# =========================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_baseline_model(source_vocab_size, target_vocab_size, embed_size=64, hidden_size=128):
    encoder = EncoderGRU(source_vocab_size, embed_size, hidden_size)
    decoder = DecoderGRU(target_vocab_size, embed_size, hidden_size)

    model = Seq2SeqBaseline(encoder, decoder, device).to(device)
    return model


def make_attention_model(source_vocab_size, target_vocab_size, embed_size=64, hidden_size=128):
    encoder = EncoderGRU(source_vocab_size, embed_size, hidden_size)
    decoder = AttentionDecoderGRU(target_vocab_size, embed_size, hidden_size)

    model = Seq2SeqAttention(encoder, decoder, device).to(device)
    return model


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate_one_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    total_time = time.time() - start_time

    return train_losses, val_losses, total_time

# =========================
# Problem 1: Baseline GRU English-to-French
# =========================

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
EMBED_SIZE = 64
HIDDEN_SIZE = 128

train_loader_en_fr, val_loader_en_fr = make_loaders(
    train_pairs,
    val_pairs,
    eng_vocab,
    fra_vocab,
    batch_size=BATCH_SIZE
)

baseline_en_fr = make_baseline_model(
    len(eng_vocab),
    len(fra_vocab),
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE
)

print("Training Problem 1 Baseline GRU English-to-French")
baseline_train_losses, baseline_val_losses, baseline_time = train_model(
    baseline_en_fr,
    train_loader_en_fr,
    val_loader_en_fr,
    epochs=EPOCHS,
    lr=LEARNING_RATE
)

baseline_acc, baseline_bleu, baseline_samples = evaluate_model(
    baseline_en_fr,
    val_pairs,
    eng_vocab,
    fra_vocab
)

print("Problem 1 Sequence Accuracy:", baseline_acc)
print("Problem 1 BLEU-4:", baseline_bleu)
print("Problem 1 Training Time:", baseline_time)
print("Problem 1 Parameters:", count_parameters(baseline_en_fr))

# =========================
# Problem 1 Loss Plot
# =========================

plt.figure(figsize=(10, 6))
plt.plot(baseline_train_losses, label="Training Loss")
plt.plot(baseline_val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Problem 1 Baseline GRU Loss Curves")
plt.legend()
plt.grid(True)
plt.savefig("results/problem1_baseline_loss.png")
plt.show()

# =========================
# Problem 1 Sample Translations
# =========================

problem1_samples_df = pd.DataFrame(baseline_samples)
problem1_samples_df

# =========================
# Problem 2: Attention GRU English-to-French
# =========================

class AttentionDecoderGRU(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super(AttentionDecoderGRU, self).__init__()

        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)

        # The attention layer uses embedded input + hidden state
        self.attention = nn.Linear(embed_size + hidden_size, MAX_LEN)

        # The GRU uses embedded input + context vector
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)

        # Final output uses GRU output + context vector
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_token, hidden, encoder_outputs):
        input_token = input_token.unsqueeze(1)

        embedded = self.embedding(input_token)

        hidden_last = hidden[-1]

        attention_input = torch.cat(
            (embedded.squeeze(1), hidden_last),
            dim=1
        )

        attention_weights = torch.softmax(
            self.attention(attention_input),
            dim=1
        )

        # Match attention length to encoder output length
        attention_weights = attention_weights[:, :encoder_outputs.shape[1]]

        attention_weights = attention_weights.unsqueeze(1)

        context = torch.bmm(attention_weights, encoder_outputs)

        gru_input = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(gru_input, hidden)

        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc(torch.cat((output, context), dim=1))

        return prediction, hidden, attention_weights.squeeze(1)


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqAttention, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(source)

        input_token = target[:, 0]

        for t in range(1, target_len):
            output, hidden, attention_weights = self.decoder(
                input_token,
                hidden,
                encoder_outputs
            )

            outputs[:, t, :] = output

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = target[:, t] if use_teacher_forcing else top1

        return outputs

  def make_attention_model(source_vocab_size, target_vocab_size, embed_size=64, hidden_size=128):
    encoder = EncoderGRU(source_vocab_size, embed_size, hidden_size)
    decoder = AttentionDecoderGRU(target_vocab_size, embed_size, hidden_size)

    model = Seq2SeqAttention(encoder, decoder, device).to(device)
    return model

attention_en_fr = make_attention_model(
    len(eng_vocab),
    len(fra_vocab),
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE
)

# =========================
# Train Fresh Attention Model
# =========================

print("Training Problem 2 Attention GRU English-to-French")

attention_train_losses, attention_val_losses, attention_time = train_model(
    attention_en_fr,
    train_loader_en_fr,
    val_loader_en_fr,
    epochs=EPOCHS,
    lr=LEARNING_RATE
)

attention_acc, attention_bleu, attention_samples = evaluate_model(
    attention_en_fr,
    val_pairs,
    eng_vocab,
    fra_vocab
)

print("Problem 2 Sequence Accuracy:", attention_acc)
print("Problem 2 BLEU-4:", attention_bleu)
print("Problem 2 Training Time:", attention_time)
print("Problem 2 Parameters:", count_parameters(attention_en_fr))

print("Training Problem 2 Attention GRU English-to-French")

attention_train_losses, attention_val_losses, attention_time = train_model(
    attention_en_fr,
    train_loader_en_fr,
    val_loader_en_fr,
    epochs=EPOCHS,
    lr=LEARNING_RATE
)

# =========================
# Problem 2 Loss Plot
# =========================

plt.figure(figsize=(10, 6))
plt.plot(attention_train_losses, label="Training Loss")
plt.plot(attention_val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Problem 2 Attention GRU Loss Curves")
plt.legend()
plt.grid(True)
plt.savefig("results/problem2_attention_loss.png")
plt.show()

# =========================
# Problem 2 Sample Translations
# =========================

problem2_samples_df = pd.DataFrame(attention_samples)
problem2_samples_df

# =========================
# Attention Map
# =========================

def plot_attention(model, sentence, source_vocab, target_vocab, title):
    prediction, attention_list = translate_sentence(
        model,
        sentence,
        source_vocab,
        target_vocab
    )

    if len(attention_list) == 0:
        print("No attention weights found.")
        return

    attention_matrix = np.array(attention_list)

    source_words = [SOS_TOKEN] + sentence.split() + [EOS_TOKEN]
    source_words = source_words[:MAX_LEN]

    target_words = prediction.split()

    plt.figure(figsize=(8, 6))
    plt.imshow(attention_matrix[:, :len(source_words)], cmap="viridis")
    plt.xticks(range(len(source_words)), source_words, rotation=45)
    plt.yticks(range(len(target_words)), target_words)
    plt.xlabel("Source Words")
    plt.ylabel("Predicted Words")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"results/{title.replace(' ', '_').lower()}.png")
    plt.show()

    print("Source:", sentence)
    print("Prediction:", prediction)

plot_attention(
    attention_en_fr,
    val_pairs[0][0],
    eng_vocab,
    fra_vocab,
    "Problem 2 Attention Map 1"
)

plot_attention(
    attention_en_fr,
    val_pairs[1][0],
    eng_vocab,
    fra_vocab,
    "Problem 2 Attention Map 2"
)

# =========================
# Problem 3: Reverse Direction French-to-English
# =========================

# Reverse the same train/validation pairs
train_pairs_fr_en = [(fra, eng) for eng, fra in train_pairs]
val_pairs_fr_en = [(fra, eng) for eng, fra in val_pairs]

train_loader_fr_en, val_loader_fr_en = make_loaders(
    train_pairs_fr_en,
    val_pairs_fr_en,
    fra_vocab,
    eng_vocab,
    batch_size=BATCH_SIZE
)

# =========================
# Problem 3: Baseline GRU French-to-English
# =========================

baseline_fr_en = make_baseline_model(
    len(fra_vocab),
    len(eng_vocab),
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE
)

print("Training Problem 3 Baseline GRU French-to-English")
baseline_fr_en_train_losses, baseline_fr_en_val_losses, baseline_fr_en_time = train_model(
    baseline_fr_en,
    train_loader_fr_en,
    val_loader_fr_en,
    epochs=EPOCHS,
    lr=LEARNING_RATE
)

baseline_fr_en_acc, baseline_fr_en_bleu, baseline_fr_en_samples = evaluate_model(
    baseline_fr_en,
    val_pairs_fr_en,
    fra_vocab,
    eng_vocab
)

print("Problem 3 Baseline Sequence Accuracy:", baseline_fr_en_acc)
print("Problem 3 Baseline BLEU-4:", baseline_fr_en_bleu)

# =========================
# Problem 3: Attention GRU French-to-English
# =========================

attention_fr_en = make_attention_model(
    len(fra_vocab),
    len(eng_vocab),
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE
)

print("Training Problem 3 Attention GRU French-to-English")
attention_fr_en_train_losses, attention_fr_en_val_losses, attention_fr_en_time = train_model(
    attention_fr_en,
    train_loader_fr_en,
    val_loader_fr_en,
    epochs=EPOCHS,
    lr=LEARNING_RATE
)

attention_fr_en_acc, attention_fr_en_bleu, attention_fr_en_samples = evaluate_model(
    attention_fr_en,
    val_pairs_fr_en,
    fra_vocab,
    eng_vocab
)

print("Problem 3 Attention Sequence Accuracy:", attention_fr_en_acc)
print("Problem 3 Attention BLEU-4:", attention_fr_en_bleu)

# =========================
# Problem 3 Loss Plots
# =========================

plt.figure(figsize=(10, 6))
plt.plot(baseline_fr_en_train_losses, label="Baseline Training Loss")
plt.plot(baseline_fr_en_val_losses, label="Baseline Validation Loss")
plt.plot(attention_fr_en_train_losses, label="Attention Training Loss")
plt.plot(attention_fr_en_val_losses, label="Attention Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Problem 3 French-to-English Loss Curves")
plt.legend()
plt.grid(True)
plt.savefig("results/problem3_reverse_loss_curves.png")
plt.show()

# =========================
# Final Results Tables
# =========================

results_summary = pd.DataFrame([
    {
        "problem": "Problem 1",
        "direction": "English-to-French",
        "model": "Baseline GRU",
        "sequence_accuracy": baseline_acc,
        "bleu_4": baseline_bleu,
        "training_time_sec": baseline_time,
        "parameters": count_parameters(baseline_en_fr)
    },
    {
        "problem": "Problem 2",
        "direction": "English-to-French",
        "model": "Attention GRU",
        "sequence_accuracy": attention_acc,
        "bleu_4": attention_bleu,
        "training_time_sec": attention_time,
        "parameters": count_parameters(attention_en_fr)
    },
    {
        "problem": "Problem 3",
        "direction": "French-to-English",
        "model": "Baseline GRU",
        "sequence_accuracy": baseline_fr_en_acc,
        "bleu_4": baseline_fr_en_bleu,
        "training_time_sec": baseline_fr_en_time,
        "parameters": count_parameters(baseline_fr_en)
    },
    {
        "problem": "Problem 3",
        "direction": "French-to-English",
        "model": "Attention GRU",
        "sequence_accuracy": attention_fr_en_acc,
        "bleu_4": attention_fr_en_bleu,
        "training_time_sec": attention_fr_en_time,
        "parameters": count_parameters(attention_fr_en)
    }
])

print(results_summary)

results_summary.to_csv("results/homework3_results_summary.csv", index=False)
problem1_samples_df.to_csv("results/problem1_sample_translations.csv", index=False)
problem2_samples_df.to_csv("results/problem2_sample_translations.csv", index=False)

problem3_baseline_samples_df = pd.DataFrame(baseline_fr_en_samples)
problem3_attention_samples_df = pd.DataFrame(attention_fr_en_samples)

problem3_baseline_samples_df.to_csv("results/problem3_baseline_samples.csv", index=False)
problem3_attention_samples_df.to_csv("results/problem3_attention_samples.csv", index=False)

# =========================
# Final BLEU and Accuracy Comparison
# =========================

plt.figure(figsize=(10, 6))
plt.bar(results_summary["model"] + "\n" + results_summary["direction"], results_summary["bleu_4"])
plt.ylabel("BLEU-4 Score")
plt.title("BLEU-4 Comparison Across Models")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("results/bleu_comparison.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(results_summary["model"] + "\n" + results_summary["direction"], results_summary["sequence_accuracy"])
plt.ylabel("Exact Match Accuracy (%)")
plt.title("Exact Match Accuracy Comparison Across Models")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("results/exact_match_accuracy_comparison.png")
plt.show()
