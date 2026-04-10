import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

VOCAB_PATH = "new_vocab.txt"
OUTPUT_DIR = "new_word_embeddings"
OUTPUT_EMB_PATH = os.path.join(OUTPUT_DIR, "mistral7b_embeddings.npy")
OUTPUT_MISSING_PATH = os.path.join(OUTPUT_DIR, "missing_words_mistral7b.txt")

MODEL_NAME = "mistralai/Mistral-7B-v0.1"


def load_vocab(vocab_path):
    vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                vocab.append(w)
    return vocab


def build_mistral_embeddings():
    print(f"Carico il dizionario da: {VOCAB_PATH}")
    vocab = load_vocab(VOCAB_PATH)
    print(f"  -> {len(vocab)} parole nel dizionario")

    print(f"Carico tokenizer e modello Mistral 7B: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  -> Uso device: {device}")

    embedding_layer = model.get_input_embeddings()
    embedding_dim = embedding_layer.embedding_dim
    print(f"  -> Dimensione embedding Mistral 7B: {embedding_dim}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    embeddings = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    missing_words = []

    unk_id = tokenizer.unk_token_id

    with torch.no_grad():
        for idx, word in enumerate(vocab):
            token_ids = tokenizer.encode(word, add_special_tokens=False)

            if (not token_ids) or (unk_id is not None and all(t == unk_id for t in token_ids)):
                missing_words.append(word)
            else:
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
                subtoken_embs = embedding_layer(token_ids_tensor)
                avg_emb = subtoken_embs.mean(dim=0)
                embeddings[idx] = avg_emb.detach().cpu().numpy()

            if (idx + 1) % 5000 == 0:
                print(f"  -> Processate {idx + 1}/{len(vocab)} parole")

    if missing_words:
        print("ATTENZIONE: alcune parole sono state mappate solo a <unk> da Mistral 7B.")
        print(f"  -> Parole mancanti: {len(missing_words)}")
        with open(OUTPUT_MISSING_PATH, "w", encoding="utf-8") as f:
            for w in missing_words:
                f.write(w + "\n")
        print(f"Lista parole mancanti: {OUTPUT_MISSING_PATH}")
    else:
        print("Tutte le parole hanno almeno un token valido in Mistral 7B ✅")

    np.save(OUTPUT_EMB_PATH, embeddings)
    print(f"Embedding Mistral 7B salvati in: {OUTPUT_EMB_PATH}")
    print(f"Forma matrice: {embeddings.shape}")


if __name__ == "__main__":
    build_mistral_embeddings()
