import os
import numpy as np
import torch
from transformers import T5Tokenizer, T5Model

# === CONFIGURAZIONE ===
VOCAB_PATH = "new_vocab.txt"
OUTPUT_DIR = "new_word_embeddings"
OUTPUT_EMB_PATH = os.path.join(OUTPUT_DIR, "t5_embeddings.npy")
OUTPUT_MISSING_PATH = os.path.join(OUTPUT_DIR, "missing_words_t5.txt")

# Puoi cambiare il modello qui, ad esempio "google/flan-t5-base"
MODEL_NAME = "t5-base"


def load_vocab(vocab_path):
    vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                vocab.append(w)
    return vocab


def build_t5_embeddings():
    # 1. Carica il dizionario
    print(f"Carico il dizionario da: {VOCAB_PATH}")
    vocab = load_vocab(VOCAB_PATH)
    print(f"  -> {len(vocab)} parole nel dizionario")

    # 2. Carica tokenizer e modello T5
    print(f"Carico tokenizer e modello T5: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5Model.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  -> Uso device: {device}")

    # Dimensione embedding (di solito = d_model)
    embedding_dim = model.config.d_model
    print(f"  -> Dimensione embedding T5 (d_model): {embedding_dim}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    embeddings = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    missing_words = []

    unk_id = tokenizer.unk_token_id

    # 3. Calcola embedding per ogni parola
    with torch.no_grad():
        for idx, word in enumerate(vocab):
            # Tokenizzazione senza special tokens
            token_ids = tokenizer.encode(word, add_special_tokens=False)

            # Parola "mancante" se la tokenizzazione è vuota
            # oppure tutti i token sono <unk>
            if (not token_ids) or all(t == unk_id for t in token_ids):
                missing_words.append(word)
                # embedding rimane il vettore di zeri
            else:
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
                # model.shared è l'embedding matrix di T5
                subtoken_embs = model.shared(token_ids_tensor)     # [seq_len, dim]
                avg_emb = subtoken_embs.mean(dim=0)                # [dim]

                embeddings[idx] = avg_emb.detach().cpu().numpy()

            # Un po' di log ogni tanto
            if (idx + 1) % 10000 == 0:
                print(f"  -> Processate {idx + 1}/{len(vocab)} parole")

    # 4. Controllo parole mancanti
    if missing_words:
        print("ATTENZIONE: alcune parole del dizionario sono state mappate solo a <unk> da T5.")
        print(f"  -> Parole mancanti: {len(missing_words)}")

        with open(OUTPUT_MISSING_PATH, "w", encoding="utf-8") as f:
            for w in missing_words:
                f.write(w + "\n")

        print(f"Lista delle parole mancanti salvata in: {OUTPUT_MISSING_PATH}")
        print("Gli embedding sono comunque salvati, ma i vettori delle parole mancanti sono tutti zero.")
    else:
        print("Tutte le parole del dizionario hanno prodotto almeno un token valido in T5 ✅")

    # 5. Salva la matrice .npy
    np.save(OUTPUT_EMB_PATH, embeddings)
    print(f"Embedding T5 salvati in: {OUTPUT_EMB_PATH}")
    print(f"Forma matrice: {embeddings.shape}")


if __name__ == "__main__":
    build_t5_embeddings()
