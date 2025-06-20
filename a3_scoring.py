import pickle
import argparse

import zstandard as zstd
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

def load_pickle_zstd(path):
    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            return pickle.load(reader)

def vendi_score_vectorized(embeddings: np.ndarray, maximum_allowed_num_items: int = None) -> float:
    # if embeddings is None or embeddings.size == 0 or embeddings.shape[0] == 0:
    #     return 1

    if maximum_allowed_num_items is not None:
        embeddings = embeddings[:maximum_allowed_num_items]

    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)

    n = embeddings.shape[0]
    sim_matrix = cosine_similarity(embeddings)

    sim_matrix = np.maximum(sim_matrix, 0)
    K_normalized = sim_matrix / n
    eigenvalues = np.linalg.eigvalsh(K_normalized)

    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    return np.exp(entropy)

def print_confidence_interval(data: np.ndarray, confidence: float = 0.95):
    if data.ndim != 1:
        raise ValueError("data must be 1D array")

    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    margin = z * (std / np.sqrt(n))

    print(f"{int(confidence*100)}% CI: {mean:.4f} +- {margin:.4f} | N={n}")

def main(args):
    emb_file_path = (
        f"{args.embedding_dir}/{args.generation_env}_{args.embedding_model_name.split('/')[-1]}_{args.generation_model_name.replace('/', '_')}.zstd"
        if not args.embedding_only_answer else
        f"{args.embedding_dir}/{args.generation_env}_{args.embedding_model_name.split('/')[-1]}_{args.generation_model_name.replace('/', '_')}_answer_only.zstd"
    )
    answer_emb_data = load_pickle_zstd(emb_file_path)

    vi_scores = np.array([
        vendi_score_vectorized(emb, args.maximum_allowed_num_items)
        for answer_emb in answer_emb_data
        for emb in answer_emb
    ])

    # for i in range(7):
    #     print_confidence_interval(vi_scores[i*100:(i+1)*100])

    print_confidence_interval(vi_scores)
    
    score_file_path = f"{args.scoring_dir}/{args.generation_env}_{args.embedding_model_name.split('/')[-1]}_{args.generation_model_name.replace('/', '_')}.csv"
    pd.DataFrame(vi_scores).to_csv(score_file_path, index=False)
    print(f"Saved: {score_file_path}")

def convert_or_none(type_fn):
    def convert(x):
        if x.lower() == "none":
            return None
        return type_fn(x)
    return convert

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--generation_env", type=str, default="standard")
    
    parser.add_argument("--embedding_model_name", type=str, default="Alibaba-NLP/gte-Qwen2-7B-instruct")
    parser.add_argument("--embedding_dir", type=str, default="embedding")
    parser.add_argument("--embedding_only_answer", action="store_true")

    parser.add_argument("--maximum_allowed_num_items", type=convert_or_none(int), default=None)

    parser.add_argument("--scoring_dir", type=str, default="scoring")

    args = parser.parse_args()
    main(args)