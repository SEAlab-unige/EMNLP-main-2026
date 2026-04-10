import os
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
from transformers import BertTokenizer, BertModel
import os

def download_and_save_bert_model(save_path='models/bert-base-uncased'):
    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.save_pretrained(save_path)

    print(f"Downloading model...")
    model = BertModel.from_pretrained('bert-base-uncased')
    model.save_pretrained(save_path)

    print(f"\n✅ BERT base uncased saved to: {save_path}")

def load_vocab(vocab_path):
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'r') as f:
        words = [line.strip() for line in f]
    print(f"Loaded {len(words)} words from vocabulary.")
    return words

def extract_embeddings(model_name, vocab_path, output_dir, output_name="bert_embeddings"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading BERT model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    vocab = load_vocab(vocab_path)
    embedding_dim = model.config.hidden_size

    embeddings = []
    found_words = []
    missing_words = []

    print("Extracting embeddings from BERT word embeddings layer...")
    for word in tqdm(vocab, desc="Extracting embeddings"):
        tokens = tokenizer.tokenize(word)
        if not tokens:
            missing_words.append(word)
            continue

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if any(id == tokenizer.unk_token_id for id in token_ids):
            missing_words.append(word)
            continue

        token_ids_tensor = torch.tensor(token_ids).to(device)
        word_embeddings = model.embeddings.word_embeddings(token_ids_tensor)
        avg_embedding = word_embeddings.mean(dim=0).detach().cpu().numpy()

        embeddings.append(avg_embedding)
        found_words.append(word)

    embeddings = np.array(embeddings)
    print(f"\n✅ Extracted embeddings for {len(found_words)} words.")
    if missing_words:
        print(f"❌ {len(missing_words)} words could not be represented in BERT vocabulary.")

    # Save output
    np.save(os.path.join(output_dir, f"{output_name}.npy"), embeddings)
    if missing_words:
        with open(os.path.join(output_dir, f"{output_name}_missing.txt"), 'w') as f:
            for word in missing_words:
                f.write(word + '\n')

    print(f"Embeddings saved in: {output_dir}")

if __name__ == '__main__':
    #download_and_save_bert_model()
    model_name = 'models/bert-base-uncased'
    vocab_path = 'new_vocab.txt'
    output_dir = 'new_word_embeddings'

    extract_embeddings(model_name, vocab_path, output_dir)
