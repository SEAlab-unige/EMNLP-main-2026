def validate_kendall_topk_inputs(A, B, k, p):
    """
    Valida gli input per la Kendall penalizzata sul top-k.
    Solleva ValueError con messaggi formali in caso di errore.
    """

    # 1) lunghezza diversa
    if len(A) != len(B):
        raise ValueError("Le permutazioni A e B hanno dimensione diversa.")

    # 2) insiemi diversi
    if set(A) != set(B):
        raise ValueError("A e B non contengono lo stesso insieme di elementi.")

    # 3) duplicati in A
    if len(set(A)) != len(A):
        raise ValueError("La permutazione A contiene elementi duplicati.")

    # 4) duplicati in B
    if len(set(B)) != len(B):
        raise ValueError("La permutazione B contiene elementi duplicati.")

    # 5) k non valido
    if not isinstance(k, int):
        raise ValueError("k non è un intero.")
    if not (1 <= k <= len(A)):
        raise ValueError("k è fuori dall'intervallo consentito.")

    # 6) p non valido
    if not isinstance(p, (int, float)):
        raise ValueError("p non è numerico.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p è fuori dall'intervallo [0, 1].")


def delta_p_for_element(x, A, B, common_topk, p):
    """
    Calcola il contributo Δ_p(x) per un elemento x dell'unione U
    nella Kendall penalizzata sul top-k.

    Parametri
    ---------
    x : int
        Elemento dell'unione U = A_k ∪ B_k.
    A, B : list[int]
        Permutazioni (liste ordinate) dello stesso insieme di elementi.
    common_topk : set[int]
        Insieme degli elementi comuni nei top-k: A_k ∩ B_k.
    p : float
        Penalità per gli elementi che non sono comuni nei top-k.

    Ritorna
    -------
    float
        Valore di Δ_p(x).
    """

    # Caso 1: x NON è nell'intersezione dei top-k → penalità p
    if x not in common_topk:
        return float(p)

    # Caso 2: x è nell'intersezione dei top-k → confrontiamo i ranking
    if A.index(x) == B.index(x):  # stesso rank nelle due permutazioni
        return 0.0
    else:  # stesso elemento ma posizione diversa nelle due permutazioni
        return 1.0


def kendall_topk_penalized(A, B, k, p):
    """
    Calcola la Kendall penalizzata sul top-k tra due permutazioni A e B.

    Parametri
    ---------
    A, B : list[int]
        Permutazioni (liste ordinate) dello stesso insieme di elementi.
    k : int
        Lunghezza del top-k da considerare.
    p : float
        Penalità per gli elementi presenti nel top-k di una sola lista.

    Ritorna
    -------
    float
        Valore di K_p(A_k, B_k).
    """

    # --- Controlli di coerenza di base ---
    validate_kendall_topk_inputs(A, B, k, p)

    # --- Costruzione dei top-k e dei set utili ---
    A_k = set(A[:k])
    B_k = set(B[:k])

    U = A_k | B_k            # unione: elementi presenti in almeno un top-k
    common_topk = A_k & B_k  # intersezione: elementi nei top-k di A e B

    # --- Somma dei contributi Δ_p(x) per ogni x in U ---
    total = 0.0
    for x in U:
        total += delta_p_for_element(x, A, B, common_topk, p)

    return total

########### SPEAR ##########
def validate_spearman_topk_inputs(A, B, k, L, gamma):
    """
    Valida gli input per la Spearman top-k con truncation e pesi.
    Solleva ValueError con messaggi formali in caso di errore.
    """

    # 1) lunghezza diversa
    if len(A) != len(B):
        raise ValueError("Le permutazioni A e B hanno dimensione diversa.")

    # 2) insiemi diversi
    if set(A) != set(B):
        raise ValueError("A e B non contengono lo stesso insieme di elementi.")

    # 3) duplicati in A
    if len(set(A)) != len(A):
        raise ValueError("La permutazione A contiene elementi duplicati.")

    # 4) duplicati in B
    if len(set(B)) != len(B):
        raise ValueError("La permutazione B contiene elementi duplicati.")

    # 5) k non valido
    if not isinstance(k, int):
        raise ValueError("k non è un intero.")
    if not (1 <= k <= len(A)):
        raise ValueError("k è fuori dall'intervallo consentito.")

    # 6) L non valido
    if not isinstance(L, int):
        raise ValueError("L non è un intero.")
    if L < k:
        raise ValueError("L è minore o uguale a k.")

    # 7) gamma non valido
    if not isinstance(gamma, (int, float)):
        raise ValueError("gamma non è numerico.")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma è fuori dall'intervallo [0, 1].")


def truncated_rank(x, perm, L):
    """
    Compute the truncated rank S(x) = min(r(x), L) for an element x
    with respect to a permutation 'perm'.

    Parameters
    ----------
    x : int
        Element whose truncated rank we want to compute.
    perm : list[int]
        Permutation (ordered list) defining the ranking.
    L : int
        Maximum rank index considered (cutoff). Any element ranked
        beyond L is assigned truncated rank L.

    Returns
    -------
    int
        Truncated rank S(x) = min(r(x), L), where r(x) is the 1-based
        rank of x in 'perm'.
    """
    try:
        # 1-based rank of the element in the permutation
        r = perm.index(x) + 1
    except ValueError:
        # x is not present in perm
        raise ValueError("The specified element is not in the permutation.")

    return min(r, L)



def weight_for_element(x, common_topk, gamma):
    """
    Calcola il peso w(x) per la Spearman top-k con pesi.

    Parametri
    ---------
    x : int
        Elemento dell'unione U dei top-k.
    common_topk : set[int]
        Insieme degli elementi comuni nei top-k (A_k ∩ B_k).
    gamma : float
        Peso da assegnare agli elementi comuni nei top-k.

    Ritorna
    -------
    float
        Peso w(x). Vale gamma se x appartiene a common_topk, altrimenti 1.0.
    """

    if x in common_topk:
        return float(gamma)
    else:
        return 1.0


def spearman_topk_penalized(A, B, k, L, gamma):
    """
    Calcola la Spearman top-k con truncation e pesi F_{L,γ}(A_k, B_k).

    Parametri
    ---------
    A, B : list[int]
        Permutazioni (liste ordinate) dello stesso insieme di elementi.
    k : int
        Lunghezza del top-k da considerare.
    L : int
        Parametro di truncation. Deve essere maggiore o uguale a k.
    gamma : float
        Peso per gli elementi comuni nei due top-k.

    Ritorna
    -------
    float
        Valore di F_{L,γ}(A_k, B_k) (versione non normalizzata).
    """

    # --- Controlli di coerenza di base ---
    validate_spearman_topk_inputs(A, B, k, L, gamma)

    # --- Costruzione dei top-k e dei set utili ---
    A_k = set(A[:k])
    B_k = set(B[:k])

    U = A_k | B_k            # unione: elementi presenti in almeno un top-k
    common_topk = A_k & B_k  # intersezione: elementi nei top-k di A e B

    # --- Somma dei contributi w(x) * |S_A(x) - S_B(x)| per ogni x in U ---
    total = 0.0
    for x in U:
        w = weight_for_element(x, common_topk, gamma)
        S_A = truncated_rank(x, A, L)
        S_B = truncated_rank(x, B, L)
        total += w * abs(S_A - S_B)


    return total


def normalize_spearman_distance(distance: float, k: int, L: int) -> float:
    """
    Normalize a Spearman top-k distance using the exact theoretical maximum
    when ranks are truncated at L and the distance is computed over
    the union U = A_k ∪ B_k.

    Parameters
    ----------
    distance : float
        Raw Spearman distance.
    k : int
        Top-k cutoff.
    L : int
        Maximum rank index considered (L >= k).

    Returns
    -------
    float
        Normalized value in [0, 1].
    """
    if not isinstance(distance, (int, float)):
        raise ValueError("distance must be numeric.")
    if distance < 0:
        raise ValueError("distance cannot be negative.")
    if not isinstance(k, int) or not isinstance(L, int):
        raise ValueError("k and L must be integers.")
    if k <= 0:
        raise ValueError("k must be strictly positive.")
    if L < k:
        raise ValueError("L must be greater than or equal to k.")

    max_distance = k * (2 * L - k - 1)
    if max_distance == 0:
        return 0.0

    return float(distance) / max_distance


def normalize_kendall_simple(distance: float, k: int) -> float:
    """
    Normalize a Kendall distance simply by dividing by k.

    Parameters
    ----------
    distance : float
        Raw Kendall distance (output of kendall_topk_penalized).
    k : int
        The top-k value used when computing the metric.

    Returns
    -------
    float
        A normalized value in the range [0, 1], assuming distance ≤ k.
    """
    if k <= 0:
        raise ValueError("k must be positive when normalizing Kendall distance.")

    return distance / (2.0 * float(k))
