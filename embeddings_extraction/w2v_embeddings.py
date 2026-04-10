import os
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm

def load_vocab(vocab_path):
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'r') as f:
        words = [line.strip() for line in f]
    print(f"Loaded {len(words)} words from vocabulary.")
    return words

def extract_embeddings(model_path, vocab_path, output_dir, output_name="w2v_embeddings"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Word2Vec model from: {model_path}")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    vocab = load_vocab(vocab_path)
    embedding_dim = model.vector_size

    embeddings = []
    found_words = []
    missing_words = []

    print("Extracting embeddings...")
    for word in tqdm(vocab, desc="Extracting embeddings"):
        if word in model:
            embeddings.append(model[word])
            found_words.append(word)
        else:
            missing_words.append(word)
            print(f"❌ Word not found in Word2Vec: {word}")

    embeddings = np.array(embeddings)
    print(f"Extracted embeddings for {len(found_words)} words.")

    # Salva i vettori e le parole corrispondenti
    np.save(os.path.join(output_dir, f"{output_name}.npy"), embeddings)
    #with open(os.path.join(output_dir, f"{output_name}_vocab.txt"), 'w') as f:
        #for word in found_words:
            #f.write(word + '\n')

    if missing_words:
        with open(os.path.join(output_dir, f"{output_name}_missing.txt"), 'w') as f:
            for word in missing_words:
                f.write(word + '\n')

    print('Missing words: ', len(missing_words))

    print(f"Embeddings saved in: {output_dir}")


if __name__ == '__main__':
    model_path = 'models/GoogleNews-vectors-negative300.bin'
    vocab_path = 'new_vocab.txt'
    output_dir = 'new_word_embeddings'

    extract_embeddings(model_path, vocab_path, output_dir)

