## Repository Description

This repository contains the reference implementation of the methodology presented in the paper:

**“Back to Words: A Model-Agnostic Comparison of Embedding Spaces through Lexical Neighborhoods”**.

The codebase implements a model-agnostic analytical framework for the comparison of heterogeneous embedding schemes, grounded in a shared lexical domain. Instead of comparing embedding spaces directly, the method operates on lexical neighborhoods, which are treated as permutations and analyzed through statistical metrics.

---

## Repository Structure

### 1. `embeddings_extraction/`

This module contains the scripts for computing embeddings and constructing lexical neighborhoods.

- **Embedding extraction scripts**  
  Each file corresponds to a specific embedding model:
  - `w2v_embeddings.py`, `glove_embeddings.py`, `fasttext_embeddings.py`
  - `bert_embeddings.py`, `roberta_embeddings.py`, `sentence_bert_embeddings.py`
  - `gpt2_embeddings.py`, `mistral7b_embeddings.py`, `qwen25_embeddings.py`
  - `mpnet.py`, `minilm_embeddings.py`, `t5_embeddings.py`, `xlnet_embeddings.py`
  - `clip_embeddings.py`

  These scripts generate vector representations for all terms in the shared vocabulary.

- **Vocabulary**  
  - `new_vocab.txt`: shared lexical domain used across all embedding models.

- **Neighborhood construction**  
  - `neighbors_indexes_full.py`: computes ordered neighborhoods for each word in the vocabulary, for every embedding model.

  The script generates a folder named `neighbors_indexes/`, where:
  - one file is created per embedding model,
  - each file contains the neighborhood permutations for all vocabulary terms.

---

### 2. `Metrica kendall + spear/`

This module implements the permutation-based comparison framework.

- `load_permutations.py`  
  Loads neighborhood data generated in the previous step.

- `permComparison.py`  
  Computes permutation-based distances between embeddings.

- `kendallSpearman.py`  
  Implements statistical metrics, including Spearman footrule and Kendall-based measures.

- `plot_permutations.py`  
  Generates visualizations such as pairwise comparison matrices and heatmaps.
