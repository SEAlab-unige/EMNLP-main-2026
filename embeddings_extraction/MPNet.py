import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# === CONFIGURAZIONE ===
VOCAB_PATH = "new_vocab.txt"
OUTPUT_DIR = "new_word_embeddings"
OUTPUT_EMB_PATH = os.path.join(OUTPUT_DIR, "mpnet_embeddings.npy")
OUTPUT_MISSING_PATH = os.path.join(OUTPUT_DIR, "missing_words_mpnet.txt")

# Modello MPNet (sentence-transformers)
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def load_vocab(vocab_path):
    vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                vocab.append(w)
    return vocab


def build_mpnet_embeddings():
    # 1. Carica il dizionario
    print(f"Carico il dizionario da: {VOCAB_PATH}")
    vocab = load_vocab(VOCAB_PATH)
    print(f"  -> {len(vocab)} parole nel dizionario")

    # 2. Carica tokenizer e modello MPNet
    print(f"Carico tokenizer e modello MPNet: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  -> Uso device: {device}")

    # Layer degli input embedding (word embedding matrix)
    embedding_layer = model.get_input_embeddings()
    embedding_dim = embedding_layer.embedding_dim
    print(f"  -> Dimensione embedding MPNet: {embedding_dim}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    embeddings = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    missing_words = []

    # ID del token sconosciuto
    unk_id = tokenizer.unk_token_id

    # 3. Calcola embedding per ogni parola
    with torch.no_grad():
        for idx, word in enumerate(vocab):
            # Tokenizzazione senza special tokens
            token_ids = tokenizer.encode(word, add_special_tokens=False)

            # Parola "mancante" se:
            # - token_ids è vuoto
            # - oppure tutti i token sono <unk>
            if (not token_ids) or all(t == unk_id for t in token_ids):
                missing_words.append(word)
                # embedding rimane il vettore di zeri
            else:
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
                # Lookup nella embedding matrix
                subtoken_embs = embedding_layer(token_ids_tensor)   # [seq_len, dim]
                avg_emb = subtoken_embs.mean(dim=0)                 # [dim]

                embeddings[idx] = avg_emb.detach().cpu().numpy()

            # Log periodico
            if (idx + 1) % 10000 == 0:
                print(f"  -> Processate {idx + 1}/{len(vocab)} parole")

    # 4. Controllo parole mancanti
    if missing_words:
        print("ATTENZIONE: alcune parole del dizionario sono state mappate solo a <unk> da MPNet.")
        print(f"  -> Parole mancanti: {len(missing_words)}")

        with open(OUTPUT_MISSING_PATH, "w", encoding="utf-8") as f:
            for w in missing_words:
                f.write(w + "\n")

        print(f"Lista delle parole mancanti salvata in: {OUTPUT_MISSING_PATH}")
        print("Gli embedding sono comunque salvati, ma i vettori delle parole mancanti sono tutti zero.")
    else:
        print("Tutte le parole del dizionario hanno prodotto almeno un token valido in MPNet ✅")

    # 5. Salva la matrice .npy
    np.save(OUTPUT_EMB_PATH, embeddings)
    print(f"Embedding MPNet salvati in: {OUTPUT_EMB_PATH}")
    print(f"Forma matrice: {embeddings.shape}")


if __name__ == "__main__":
    build_mpnet_embeddings()
