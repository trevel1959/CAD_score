import json
import argparse
import os
import pickle

import zstandard as zstd
import numpy as np

from openai import OpenAI
from collections import defaultdict

def download_batch_result(args):
    # --- Find batch ID ---
    with open(f"{args.batch_input_dir}/batch_info.json", "r") as f:
        batches = json.load(f)

    input_file_name = f"{args.generation_env}_{args.embedding_model_name}_{args.generation_model_name.replace('/', '_')}.jsonl"

    print(f"Searching for batch ID for {input_file_name}...")
    batch_id = None
    for batch in batches:
        if batch["input_file_name"] == input_file_name:
            print(f"Found batch ID: {batch['batch_id']}")
            batch_id = batch["batch_id"]
            break
    if batch_id is None:
        raise ValueError(f"Batch ID not found for input file: {input_file_name}")

    # --- Download batch result ---
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        raise ValueError(f"Batch {batch_id} is not completed")

    batch_result_text = client.files.content(batch.output_file_id).text
    
    # --- Parse batch result ---
    embedding_results = []
    for line in batch_result_text.splitlines():
        if line.strip():
            result = json.loads(line)
            embedding_results.append(result)

    nested = defaultdict(lambda: defaultdict(list))

    for res in embedding_results:
        _, t_str, q_str, _ = res["custom_id"].split("-")
        t, q = int(t_str), int(q_str)
        emb = res["response"]["body"]["data"][0]["embedding"]
        nested[t][q].append(emb)

    grouped = [
        [ nested[t][q] for q in sorted(nested[t].keys()) ]
        for t in sorted(nested.keys())
    ]

    # --- Save embeddings ---
    output_file = (
        f"{args.embedding_dir}/{args.generation_env}_{args.embedding_model_name}_{args.generation_model_name.replace('/', '_')}.zstd"
        if not args.embedding_only_answer else
        f"{args.embedding_dir}/{args.generation_env}_{args.embedding_model_name}_{args.generation_model_name.replace('/', '_')}_answer_only.zstd"
    )
    with open(output_file, "wb") as f:
        cctx = zstd.ZstdCompressor(level=5)
        with cctx.stream_writer(f) as writer:
            pickle.dump(grouped, writer, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Downloaded embeddings to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--generation_env", type=str, default="standard")
    parser.add_argument("--embedding_model_name", type=str, default="text-embedding-3-large")
    parser.add_argument("--embedding_dir", type=str, default="embedding")
    parser.add_argument("--batch_input_dir", type=str, default="openai_batch_input")
    parser.add_argument("--embedding_only_answer", action="store_true")

    args = parser.parse_args()
    download_batch_result(args)
