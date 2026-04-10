import os
import numpy as np
from tqdm import tqdm

def load_vocab(vocab_path):
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'r') as f:
        words = [line.strip() for line in f]
    print(f"Loaded {len(words)} words from vocabulary.")
    return words

def load_glove_model(glove_path):
    print(f"Loading GloVe model from: {glove_path}")
    glove_model = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe vectors"):
            split = line.strip().split()
            word = split[0]
            vector = np.array(split[1:], dtype=np.float32)
            glove_model[word] = vector
    print(f"Loaded {len(glove_model)} GloVe vectors.")
    return glove_model

def extract_embeddings(glove_path, vocab_path, output_dir, output_name="glove_embeddings"):
    os.makedirs(output_dir, exist_ok=True)

    glove_model = load_glove_model(glove_path)
    vocab = load_vocab(vocab_path)
    embedding_dim = len(next(iter(glove_model.values())))

    embeddings = []
    found_words = []
    missing_words = []

    print("Extracting embeddings...")
    for word in tqdm(vocab, desc="Extracting embeddings"):
        if word in glove_model:
            embeddings.append(glove_model[word])
            found_words.append(word)
        else:
            missing_words.append(word)
            print(word)

    embeddings = np.array(embeddings)
    print(f"\n✅ Found embeddings for {len(found_words)} words.")
    if missing_words:
        print(f"❌ {len(missing_words)} words not found in GloVe.")

    # Salvataggio output
    np.save(os.path.join(output_dir, f"{output_name}.npy"), embeddings)
    #with open(os.path.join(output_dir, f"{output_name}_vocab.txt"), 'w') as f:
        #for word in found_words:
            #f.write(word + '\n')
    if missing_words:
        with open(os.path.join(output_dir, f"{output_name}_missing.txt"), 'w') as f:
            for word in missing_words:
                f.write(word + '\n')

    print(f"Embeddings saved in: {output_dir}")

if __name__ == '__main__':
    glove_path = 'models/glove.6B.300d.txt'  # o altro modello GloVe
    vocab_path = 'new_vocab.txt'
    output_dir = 'new_word_embeddings'

    extract_embeddings(glove_path, vocab_path, output_dir)
