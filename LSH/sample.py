# coding:utf-8
# Author: Knight
# Date: 14/08/2025

import requests
import pandas as pd
import numpy as np
import io
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

def build_shingle(sent: str, k: int) -> set[str]:
    if k <= 0:
        raise ValueError(f"k must be a positive integer, however get: {k}")
    n = len(sent)
    if k > n:
        return set()
    return {sent[i : i + k] for i in range(n - k + 1)}

def build_vocab(shingles_set: list[set[str]]) -> dict[str, int]:
    full_set = {item for shingles in shingles_set for item in shingles}
    return {shingle : i for i, shingle in enumerate(sorted(full_set))}

def get_one_hot_encoding(shingles: set[str], vocab: dict[str, int]) -> np.ndarray:
    encodings = np.zeros(len(vocab), dtype=np.uint8)
    for shingle in shingles:
        idx = vocab.get(shingle)
        if idx is not None:
            encodings[idx] = 1
    return encodings

def create_minhash_funcs(one_hot_dim: int, sig_dim: int) -> np.ndarray:
    rng = np.random.default_rng(17)
    minhash_funcs = np.empty((sig_dim, one_hot_dim), dtype=np.int32)
    base = np.arange(one_hot_dim, dtype=np.int32) + 1
    for i in range(sig_dim):
        minhash_funcs[i, :] = rng.permutation(base)
    return minhash_funcs

def get_signature(minhash_funcs: np.ndarray, one_hot_encoding: np.ndarray) -> np.ndarray:
    non_zero_col_idx = np.nonzero(one_hot_encoding)[0]
    if non_zero_col_idx.size == 0:
        raise ValueError("Signature requested for an empty set (no shingles present).")
    signature = np.min(minhash_funcs[: , non_zero_col_idx], axis=1)
    return signature.astype(np.int32)

class LSH:
    
    def __init__(self, band: int):
        if band <= 0:
            raise ValueError(f"band must be a positive integer, however get: {band}")
        self.band = band
        self.buckets: list[dict[tuple, list[int]]] = [dict() for _ in range(self.band)]
    
    def split_signature(self, signature: np.ndarray) -> np.ndarray:
        sig_dim = int(signature.shape[0])
        assert len(signature) % self.band == 0, ValueError(f"Signature dimension ({sig_dim}) must be divisible by band ({self.band}).")
        r = len(signature) // self.band
        return np.stack([signature[i : i + r] for i in range(0, sig_dim, r)])

    def add_signature(self, signature: np.ndarray, unique_id: int):
        subvecs = self.split_signature(signature)
        for i, vec in enumerate(subvecs):
            key = tuple(np.asarray(vec, dtype=int).tolist())
            self.buckets[i].setdefault(tuple(vec.tolist()), []).append(unique_id)
    
    def find_candidates_pairs(self) -> set[tuple[int, int]]:
        candidates = []
        for bucket in self.buckets:
            for ids in bucket.values():
                if len(ids) > 1:
                    candidates.extend(combinations(ids, 2))
        return set(candidates)

    def find_similar(self, signature: np.ndarray) -> set[int]:
        subvects = self.split_signature(signature)
        similar_idx = []
        for i, vec in enumerate(subvects):
            key = tuple(np.asarray(vec, dtype=int).tolist())
            hit = self.buckets[i].get(key)
            if hit:
                similar_idx.extend(hit)
        return set(similar_idx)

def jaccard(a: set, b: set):
    return len(a.intersection(b)) / len(a.union(b))

def probability(s, r, b):
    return 1 - (1 - s**r)**b

def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

if __name__ == "__main__":

    url = "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt"

    text = requests.get(url).text

    """
    Dataset Information:
        Columns 

            pair_ID – A unique ID for each sentence pair.

            sentence_A – The first sentence in the pair.

            sentence_B – The second sentence in the pair.

            relatedness_score – A human-annotated score (0–5) of how semantically similar the two sentences are:

                0 → completely unrelated

                5 → identical meaning

                Decimals like 3.2, 4.7 reflect partial similarity.

            entailment_judgment – A categorical label for the entailment relationship:

                ENTAILMENT → Sentence A implies Sentence B.

                CONTRADICTION → They conflict in meaning.

                NEUTRAL → No implication either way.
    """
    data = pd.read_csv(io.StringIO(text), sep='\t')

    sentences = data["sentence_A"].to_list()

    shingles_set = []
    for sent in sentences:
        shingles_set.append(build_shingle(sent, k=8))
    
    vocab = build_vocab(shingles_set)

    one_hot_encodings = []
    for shingle in shingles_set:
        one_hot_encodings.append(get_one_hot_encoding(shingle, vocab))
    one_hot_encodings = np.stack(one_hot_encodings)

    minhash_funcs = create_minhash_funcs(len(vocab), 100)
    signature_encodings = []
    for one_hot in one_hot_encodings:
        signature_encodings.append(get_signature(minhash_funcs, one_hot))
    signature_encodings = np.stack(signature_encodings)

    lsh = LSH(25)
    for i, sig in enumerate(signature_encodings):
        lsh.add_signature(sig, i)
    candidates_pairs = lsh.find_candidates_pairs()

    cand_idx = lsh.find_similar(signature_encodings[15])
    print("Ori: ", sentences[15])
    for idx in cand_idx:
        print(f"Sim_{idx}: ", sentences[idx])

    # ----------------- plot different band -----------------
    import matplotlib.pyplot as plt
    import seaborn as sns
    b_vals = [100, 50, 25, 20, 10, 5, 2]
    s_scores = np.arange(0.01, 1, 0.01)
    probs = []
    for b in b_vals:
        r = int(100 / b)
        p_scores = [probability(s, r, b) for s in s_scores]
        probs.append(
            pd.DataFrame({
                "p": p_scores,
                "s": s_scores,
                "b": str(b),
            })
        )
    probs = pd.concat(probs, ignore_index=True)
    sns.lineplot(data=probs, x='s', y='p', hue='b')

    # ----------------- plot data -----------------
    data_points = []
    total_data_num = one_hot_encodings.shape[0]
    chosen = set()
    sample_size = 50_000
    for _ in range(sample_size):
        x, y = np.random.choice(total_data_num, 2)
        if x == y or (x, y) in chosen: continue
        chosen.add((x, y))

        sig_x = signature_encodings[x]
        sig_y = signature_encodings[y]
        candidate = 1 if (x, y) in candidates_pairs else 0
        cosine = cosine_similarity([sig_x], [sig_y])[0][0]

        data_points.append(
            {
                "x": x,
                "y": y,
                "jaccard": jaccard(set(sig_x), set(sig_y)),
                "cosine": cosine,
                "candidate": candidate,
            }
        )

    data_points = pd.DataFrame(data_points)
    sns.scatterplot(data=data_points, x="jaccard", y="candidate", alpha=0.5, color="k")
    plt.show()

    # rows= []
    # data_len = one_hot_encodings.shape[0]
    # chosen = set()
    # # take random sample of pairs
    # sample_size = 50_000
    # for _ in range(sample_size):
    #     x, y = np.random.choice(data_len, 2)
    #     if x == y or (x, y) in chosen: continue
    #     chosen.add((x, y))
    #     vector_x = signature_encodings[x]
    #     vector_y = signature_encodings[y]
    #     candidate = 1 if (x, y) in candidates_pairs else 0
    #     cosine = cosine_similarity([vector_x], [vector_y])[0][0]
    #     rows.append({
    #             'x': x,
    #             'y': y,
    #             'jaccard': jaccard(set(vector_x), set(vector_y)),
    #             'cosine': cosine,
    #             'candidate': candidate
    #     })

    # pairs = pd.DataFrame(rows)
    # # add a normalized cosine column for better alignment
    # cos_min = pairs['cosine'].min()
    # cos_max = pairs['cosine'].max()
    # pairs['cosine_norm'] = (pairs['cosine'] - cos_min) / (cos_max - cos_min)