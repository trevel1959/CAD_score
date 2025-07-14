"""
Other Text Generation Evaluation Metrics
"""

import argparse
import gzip
import json
from typing import List, Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def _tokenize(sentences: List[str], lowercase: bool = True) -> List[List[str]]:
    tokenized = []
    for sentence in sentences:
        if lowercase:
            sentence = sentence.lower()
        tokenized.append(nltk.word_tokenize(sentence))
    return tokenized

def _bleu_weights(max_n: int) -> Tuple[float, ...]:
    return tuple([1.0 / max_n] * max_n)

def score_self_bleu(sentences: List[str], max_n: int = 4, smoothing: bool = True) -> float:
    """Calculate average self-BLEU score for a list of sentences."""
    tokenized = _tokenize(sentences)
    n = len(tokenized)
    
    if n < 2:
        return 0.0

    weights = _bleu_weights(max_n)
    smooth_function = SmoothingFunction().method1 if smoothing else None

    scores = []
    for i in range(n):
        hypothesis = tokenized[i]
        references = [tokenized[j] for j in range(n) if j != i]
        score = sentence_bleu(
            references, 
            hypothesis, 
            weights=weights, 
            smoothing_function=smooth_function
        )
        scores.append(score)

    return float(np.mean(scores))


def _compress_len_gzip(data: bytes) -> int:
    return len(gzip.compress(data))

def _ncd_gzip(x: str, y: str) -> float:
    x_bytes = x.encode("utf-8")
    y_bytes = y.encode("utf-8")
    
    cx = _compress_len_gzip(x_bytes)
    cy = _compress_len_gzip(y_bytes)
    cxy = _compress_len_gzip(x_bytes + y_bytes)
    
    return (cxy - min(cx, cy)) / max(cx, cy)

def score_ncd(sentences: List[str], arg_n: int = None) -> float:
    """Calculate average compression distance for a list of sentences."""
    
    n = len(sentences)
    if n == 0:
        return np.empty((0, 0), dtype=float)

    matrix = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            distance = _ncd_gzip(sentences[i], sentences[j])
            matrix[i, j] = matrix[j, i] = distance

    if matrix.shape[0] < 2:
        return 0.0
    
    upper_tri = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return float(upper_tri.mean())


def _load_data(generation_dir: str, generation_env: str, generation_model_name: str) -> List:
    input_path = f"{generation_dir}/{generation_env}_{generation_model_name.replace('/', '_')}.json"
    
    with open(input_path, "r") as f:
        return json.load(f)


def _calculate_scores(data: List, scoring_method: str, arg_n: int = None) -> np.ndarray:
    scoring_functions = {
        "ncd": score_ncd,
        "self_bleu": score_self_bleu,
    }

    if scoring_method not in scoring_functions:
        raise ValueError(f"Unknown scoring method: {scoring_method}")
    
    if scoring_method in ["self_bleu"] and arg_n is None:
        raise ValueError("arg_n is required for self_bleu scoring method")

    scoring_func = scoring_functions[scoring_method]
    return np.array([
        scoring_func(item, arg_n)
        for task in data
        for item in task
    ])


def _save_results(scores: np.ndarray, scoring_dir: str, generation_env: str, scoring_method: str, generation_model_name: str, arg_n: int = None) -> str:
    _scoring_method = scoring_method
    if arg_n is not None:
        _scoring_method += f"_{arg_n}"

    output_path = f"{scoring_dir}/{generation_env}_{_scoring_method}_{generation_model_name.replace('/', '_')}.csv"
    pd.DataFrame(scores).to_csv(output_path, index=False)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Evaluate text generation using various metrics")
    
    parser.add_argument("--generation_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--generation_dir", type=str, default="generation")
    parser.add_argument("--generation_env", type=str, default="standard_full")

    parser.add_argument("--scoring_dir", type=str, default="scoring")
    parser.add_argument("--scoring_method", type=str, default="ncd")
    parser.add_argument("--arg_n", type=int, default=None)

    args = parser.parse_args()

    data = _load_data(args.generation_dir, args.generation_env, args.generation_model_name)
    scores = _calculate_scores(data, args.scoring_method, args.arg_n)
    output_path = _save_results(scores, args.scoring_dir, args.generation_env, args.scoring_method, args.generation_model_name, args.arg_n)
     
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()