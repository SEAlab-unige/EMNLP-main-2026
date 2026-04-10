import os
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

# ===================== PARAMETRI =====================
VOCAB_PATH = "new_vocab.txt"  # una parola per riga, NELL'ORDINE degli embedding
EMBEDDINGS = {
    "w2v": "new_word_embeddings/w2v_embeddings.npy",
    "glove": "new_word_embeddings/glove_embeddings.npy",
    "fasttext": "new_word_embeddings/fasttext_embeddings.npy",
    "bert": "new_word_embeddings/bert_embeddings.npy",
    "roberta": "new_word_embeddings/roberta_embeddings.npy",
    "clip": "new_word_embeddings/clip_embeddings.npy",
    "gpt-2": "new_word_embeddings/gpt2_embeddings.npy",
    "minilm": "new_word_embeddings/minilm_embeddings.npy",
    "mistral": "new_word_embeddings/mistral7b_embeddings.npy",
    "mpnet": "new_word_embeddings/mpnet_embeddings.npy",
    "qwen25": "new_word_embeddings/qwen25_embeddings.npy",
    "sentence_bert": "new_word_embeddings/sentence_bert_embeddings.npy",
    "t5": "new_word_embeddings/t5_embeddings.npy",
    "xlnet": "new_word_embeddings/xlnet_embeddings.npy"
}

OUTPUT_DIR = "neighbors_indexes"
CHUNK_SIZE = 65536
SELF_DISTANCE = np.inf
# =====================================================


# ---------- vocabolario ----------
def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab = [ln.strip() for ln in f if ln.strip()]
    if not vocab:
        raise ValueError("vocab.txt vuoto o illeggibile.")
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx


# ---------- Utilità ----------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_id_mapping(path, words):
    """Salva ID→parola (1-based) per riferimento umano."""
    with open(path, "w", encoding="utf-8") as f:
        for i, w in enumerate(words, start=1):
            f.write(f"{i}\t{w}\n")


# ---------- distanza euclidea quadrata ----------
def compute_and_save_single_binary_euclidean_sq(emb_path: str, words: list, out_root: str):
    """
    Crea un unico file .npy (memmap) con dtype:
      [('id', np.uint32), ('dist2', np.float32)]
    shape: (n, n-1)

    La riga i contiene TUTTI i vicini di i (self escluso),
    ordinati per distanza euclidea al quadrato (crescente).
    """
    n = len(words)
    E = np.load(emb_path)  # shape (n, d)

    if E.ndim != 2:
        raise ValueError(f"Embeddings devono essere 2D (n x d). Trovato: {E.shape}.")
    if E.shape[0] != n:
        raise ValueError(f"Righe embeddings ({E.shape[0]}) != parole del dizionario ({n}).")

    # Pre-calcola le norme al quadrato
    norms2_all = np.sum(E * E, axis=1)  # shape (n,)

    # Prepara output
    ensure_dir(out_root)
    out_path = os.path.join(out_root, "euclidean_sq_neighbors_all_pairs.npy")
    ids_map_path = os.path.join(out_root, "ids_map.txt")
    save_id_mapping(ids_map_path, words)

    dtype_struct = np.dtype([('id', np.uint32), ('dist2', np.float32)])
    mm = open_memmap(out_path, mode='w+', dtype=dtype_struct, shape=(n, n - 1))

    for start in tqdm(
        range(0, n, CHUNK_SIZE),
        total=(n + CHUNK_SIZE - 1) // CHUNK_SIZE,
        desc=f"Computing euclidean^2 for {os.path.basename(emb_path)}"
    ):
        end = min(start + CHUNK_SIZE, n)

        # (B, d) @ (d, n) -> (B, n)
        dots = E[start:end] @ E.T
        norms2_block = norms2_all[start:end][:, None]  # (B, 1)

        # d^2 = ||x||^2 + ||y||^2 - 2 x·y
        d2 = norms2_block + norms2_all[None, :] - 2.0 * dots

        # Evita piccoli negativi numerici
        np.maximum(d2, 0.0, out=d2)

        # Escludi self
        for local_i, global_i in enumerate(range(start, end)):
            d2[local_i, global_i] = SELF_DISTANCE

        # Ordina in ordine crescente di distanza
        order = np.argsort(d2, axis=1)

        # self va in fondo perché inf, quindi prendiamo tutto tranne l'ultima colonna
        ids_sorted = (order[:, :-1].astype(np.uint32) + 1)  # 1-based
        d2_sorted = d2[np.arange(end - start)[:, None], order[:, :-1]].astype(np.float32)

        mm['id'][start:end, :] = ids_sorted
        mm['dist2'][start:end, :] = d2_sorted

    del mm  # flush su disco
    return out_path, ids_map_path


# ---------- main per embedder ----------
def process_embedder(name, emb_path, words):
    embedder_out_dir = os.path.join(OUTPUT_DIR, name)
    ensure_dir(embedder_out_dir)

    print(f"[INFO] [{name}] Embedding: {emb_path}")
    print(f"[INFO] [{name}] Output dir: {embedder_out_dir}")

    out_path, ids_map_path = compute_and_save_single_binary_euclidean_sq(
        emb_path=emb_path,
        words=words,
        out_root=embedder_out_dir
    )

    print(f"[OK] [{name}] Neighbor file: {out_path}")
    print(f"[OK] [{name}] ID map       : {ids_map_path}")


def main():
    vocab, word2idx = load_vocab(VOCAB_PATH)

    for name, path in EMBEDDINGS.items():
        if not os.path.exists(path):
            print(f"[WARN] File embedding mancante: {path}")
            continue

        try:
            process_embedder(name, path, vocab)
        except Exception as e:
            print(f"[ERRORE] [{name}] {e}")


if __name__ == "__main__":
    main()