import os
import numpy as np
from tqdm import tqdm
from gensim.models.fasttext import load_facebook_vectors

def load_vocab(vocab_path):
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'r') as f:
        words = [line.strip() for line in f]
    print(f"Loaded {len(words)} words from vocabulary.")
    return words

def extract_embeddings(ft_model_path, vocab_path, output_dir, output_name="fasttext_embeddings"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading FastText model from: {ft_model_path}")
    model = load_facebook_vectors(ft_model_path)

    vocab = load_vocab(vocab_path)
    embedding_dim = model.vector_size

    embeddings = []
    found_words = []
    missing_words = []

    print("Extracting embeddings (FastText handles OOV)...")
    for word in tqdm(vocab, desc="Extracting embeddings"):
        try:
            vector = model[word]
            embeddings.append(vector)
            found_words.append(word)
        except KeyError:
            missing_words.append(word)  # Questo non dovrebbe accadere con FastText

    embeddings = np.array(embeddings)
    print(f"\n✅ Extracted embeddings for {len(found_words)} words.")
    if missing_words:
        print(f"❌ {len(missing_words)} words not found in FastText.")

    # Salvataggio output
    np.save(os.path.join(output_dir, f"{output_name}.npy"), embeddings)
    if missing_words:
        with open(os.path.join(output_dir, f"{output_name}_missing.txt"), 'w') as f:
            for word in missing_words:
                f.write(word + '\n')

    print(f"Embeddings saved in: {output_dir}")

if __name__ == '__main__':
    ft_model_path = 'models/cc.en.300.bin'  # modello FastText binario
    vocab_path = 'new_vocab.txt'
    output_dir = 'new_word_embeddings'

    extract_embeddings(ft_model_path, vocab_path, output_dir)
