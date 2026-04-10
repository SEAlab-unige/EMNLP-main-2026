"""
Module that loads permutation data from multiple text files into a structured dictionary.
"""
#---------------------- Imports ------------------------------
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Type alias for better readability: dict[id][method] = permutation
PermutationsDict = Dict[int, Dict[str, List[int]]]
# Streaming yield type: (id, {method: permutation})
PermutationRecord = Tuple[int, Dict[str, List[int]]]

# Map of method names to their corresponding filenames
PERMUTATION_FILES = {
    "bert": "bert_neighbors_indices_full.txt",
    "glove": "glove_neighbors_indices_full.txt",
    "w2v": "w2v_neighbors_indices_full.txt",
    "fasttext": "fasttext_neighbors_indices_full.txt",
    "roberta": "roberta_neighbors_indices_full.txt",
    "clip": "clip_neighbors_indices_full.txt",
    "gpt-2": "gpt-2_neighbors_indices_full.txt",
    "minilm": "minilm_neighbors_indices_full.txt",
    "mistral": "mistral_neighbors_indices_full.txt",
    "mpnet": "mpnet_neighbors_indices_full.txt",
    "qwen25": "qwen25_neighbors_indices_full.txt",
    "sentence_bert": "sentence_bert_neighbors_indices_full.txt",
    "t5": "t5_neighbors_indices_full.txt",
    "xlnet": "xlnet_neighbors_indices_full.txt",
}

""" PERMUTATION_FILES = {
    "bert": "bert_neighbors_indices_full.txt",
    "glove": "glove_neighbors_indices_full.txt",
    "w2v": "w2v_neighbors_indices_full.txt",
    "fasttext": "fasttext_neighbors_indices_full.txt",
    "roberta": "roberta_neighbors_indices_full.txt",
} """

def parse_permutation_line(line: str) -> tuple[int, List[int]]:
    """
    Parse a single line of permutation file.
    Each line is expected to start with an integer ID followed by a sequence of integers representing the permutation.   
    Parameters
    ----------
    line : str
        A line from the permutation file.                           
        Returns
        -------
        tuple[int, List[int]]
        A tuple containing the ID and the corresponding permutation list.
    """

    tokens = line.strip().split() # Split by whitespace
    if not tokens:
        raise ValueError("Error: Empty line encountered.")

    numbers = [int(tok) for tok in tokens] # Convert tokens to integers
    id_ = numbers[0]
    permutation = numbers[1:]

    if not permutation:
        raise ValueError(f"No permutation found for id {id_}: the line contains only one number.")

    return id_, permutation


def load_permutations_from_file(
    filepath: Path,
    method: str,
    permutations: PermutationsDict,
) -> None:
    """
    Reads a permutation file and populates the permutations dictionary.
    Each line in the file should start with an integer ID followed by a sequence of integers representing the permutation.
    Parameters
    ----------
    filepath : Path
        Path to the permutation file.
    method : str
        The method name associated with the permutations in this file.
    permutations : PermutationsDict
        The dictionary to populate with the loaded permutations.
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If there are duplicate IDs for the same method or if a line is malformed.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1): # Read line by line with line numbers
            # Skip empty lines
            if not line.strip():
                continue

            try:
                id_, permutation = parse_permutation_line(line) # Parse the line
            except ValueError as e:
                raise ValueError(f"Error at line {line_number} in file {filepath}: {e}")

            # If the id is not yet present, create the sub-dictionary
            if id_ not in permutations:
                permutations[id_] = {}

            # Optional check: if the same method for the same id is already present
            if method in permutations[id_]:
                raise ValueError(
                    f"Duplicate ID for method '{method}' (id={id_}) in file {filepath}, line {line_number}"
                )

            permutations[id_][method] = permutation


def load_all_permutations(
    base_dir: str = ".",
    files_map: Dict[str, str] = None,
) -> PermutationsDict:
    """
    Loads permutations from multiple files into a structured dictionary.
    Parameters
    ----------
    base_dir : str              
        Base directory where the permutation files are located.
    files_map : Dict[str, str], optional
        A mapping of method names to their corresponding filenames. If None, the default PERMUTATION_FILES is used.
    Returns
    -------
    PermutationsDict
        A dictionary where each key is an integer ID and the value is another dictionary mapping method names to their corresponding permutation lists.
    Raises
    ------
    FileNotFoundError
        If any of the specified files do not exist.
    ValueError
        If there are duplicate IDs for the same method or if a line is malformed.
    """
    if files_map is None:
        files_map = PERMUTATION_FILES

    base_path = Path(base_dir)
    permutations: PermutationsDict = {}
    print(f"Loading permutations from: {base_path.resolve()}")
    print(f"Files to load: {len(files_map)}")

    for method, filename in files_map.items():
        filepath = base_path / filename
        print(f"Loading method '{method}' from: {filepath}")
        load_permutations_from_file(filepath, method, permutations)
        print(f"Finished method '{method}'.")

    return permutations


def stream_aligned_permutations(
    base_dir: str = ".",
    files_map: Dict[str, str] = None,
) -> Iterable[PermutationRecord]:
    """
    Stream permutations one ID at a time, assuming all files have
    the same IDs in the same order.

    This avoids loading the entire dataset into memory. For each ID it yields
    (id, {method: permutation}) exactly like load_all_permutations would
    produce for a single entry.
    """
    if files_map is None:
        files_map = PERMUTATION_FILES

    base_path = Path(base_dir)
    methods = list(files_map.keys())

    # Prepare file handles up-front so we can read each line in lockstep.
    file_handles = {}
    for method, filename in files_map.items():
        filepath = base_path / filename
        if not filepath.is_file():
            raise FileNotFoundError(f"File not found: {filepath}")
        file_handles[method] = filepath.open("r", encoding="utf-8")

    try:
        # zip_longest lets us detect mismatched line counts (one file ends early).
        for line_idx, lines in enumerate(
            zip_longest(*file_handles.values()), start=1
        ):
            # None means one of the files had fewer lines than the others.
            if any(line is None for line in lines):
                missing = [
                    m
                    for m, line in zip(methods, lines)
                    if line is None
                ]
                raise ValueError(
                    f"Line count mismatch at line {line_idx}: "
                    f"files for methods {missing} ended early."
                )

            current_id = None
            parsed = {}

            for method, line in zip(methods, lines):
                if not line.strip():
                    raise ValueError(
                        f"Empty line encountered at line {line_idx} in file "
                        f"{files_map[method]}"
                    )

                id_, permutation = parse_permutation_line(line)

                if current_id is None:
                    current_id = id_
                elif id_ != current_id:
                    raise ValueError(
                        f"ID mismatch at line {line_idx}: method '{method}' "
                        f"has id {id_} while others have id {current_id}."
                    )

                parsed[method] = permutation

            yield current_id, parsed

        # After zip_longest is exhausted, check that no file still has extra data.
        for method, fh in file_handles.items():
            remainder = fh.readline()
            if remainder:
                raise ValueError(
                    f"File for method '{method}' has extra lines after "
                    f"synchronized reading."
                )
    finally:
        for fh in file_handles.values():
            fh.close()


if __name__ == "__main__":
    # Esempio di uso: carica tutti i file nella directory corrente
    perms = load_all_permutations("../neighbors_indexes/")

    print(f"Numero totale di ID caricati: {len(perms)}")
    # Mostriamo solo qualche esempio
    sample_ids = list(perms.keys())[:3]
    for id_ in sample_ids:
        print(f"\nID: {id_}")
        for method, permutation in perms[id_].items():
            print(f"  Metodo: {method}, lunghezza permutazione: {len(permutation)}")
            # Se vuoi vedere l'inizio della permutazione:
            print(f"    primi 10 elementi: {permutation[:10]}")
