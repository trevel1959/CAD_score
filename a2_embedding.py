import json 
import torch
import argparse
import pickle
import logging
import tqdm

import zstandard as zstd
import numpy as np
import pandas as pd

from vllm import LLM

def calculate_answer_embedding(args):
    # --- Load dataset and generated answers ---
    dataset_file_path = f"{args.dataset_dir}/{args.dataset_file}"
    with open(dataset_file_path, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    gen_file_path = f"{args.generation_dir}/{args.generation_env}_{args.generation_model_name.replace('/', '_')}.json"
    with open(gen_file_path, "r", encoding="utf-8") as f:
        answer_data = json.load(f)

    # --- Construct texts for embedding ---
    texts_answer_with_query = []
    answer_counts = []
    for task, answers_per_task in zip(query_data, answer_data):
        per_task_counts = []
        for query, answers_per_query in zip(task["task_list"], answers_per_task):
            per_task_counts.append(len(answers_per_query))
            for answer in answers_per_query:
                if args.embedding_only_answer:
                    texts_answer_with_query.append(answer)
                else:
                    texts_answer_with_query.append(f"Q: {task['task_prompt']} {query}\nA: {answer}")
        answer_counts.append(per_task_counts)

    # --- Load embedding model ---
    model = LLM(
        model=args.embedding_model_name,
        tokenizer=args.embedding_model_name,
        tokenizer_mode="auto",
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        dtype="half",
        task="embed"
    )

    # --- Embed texts ---
    embs_answer_with_query = np.stack([np.array(e.outputs.embedding) for e in model.embed(texts_answer_with_query)])

    # --- Reshape and save answer embeddings ---
    reshaped_embs = []
    idx = 0
    for per_task_counts in answer_counts:
        task_embs = []
        for num_ans in per_task_counts:
            task_embs.append(embs_answer_with_query[idx:idx+num_ans])
            idx += num_ans
        reshaped_embs.append(task_embs)

    emb_file_path = (
        f"{args.embedding_dir}/{args.generation_env}_{args.embedding_model_name.split('/')[-1]}_{args.generation_model_name.replace('/', '_')}.zstd"
        if not args.embedding_only_answer else
        f"{args.embedding_dir}/{args.generation_env}_{args.embedding_model_name.split('/')[-1]}_{args.generation_model_name.replace('/', '_')}_answer_only.zstd"
    )
    with open(emb_file_path, "wb") as f:
        cctx = zstd.ZstdCompressor(level=5)
        with cctx.stream_writer(f) as writer:
            pickle.dump(reshaped_embs, writer, protocol=pickle.HIGHEST_PROTOCOL)

    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--dataset_file", type=str, default="creative_task_sample.json")

    parser.add_argument("--generation_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--generation_dir", type=str, default="generation")
    parser.add_argument("--generation_env", type=str, default="standard")

    parser.add_argument("--embedding_model_name", type=str, default="Alibaba-NLP/gte-Qwen2-7B-instruct")
    parser.add_argument("--embedding_dir", type=str, default="embedding")
    parser.add_argument("--embedding_only_answer", action="store_true")
    args = parser.parse_args()
    
    logging.getLogger("vllm").setLevel(logging.ERROR)
    calculate_answer_embedding(args)