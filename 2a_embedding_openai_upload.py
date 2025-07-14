import json 
import torch
import argparse
import pickle
import logging
import tqdm

import zstandard as zstd
import numpy as np
import pandas as pd

from openai import OpenAI

def gpt_emb_batch(dataset_name, index, input, embedding_model_name):
    return {
        "custom_id": f"{dataset_name}-{index}",
        "method": "POST",
        "url": "/v1/embeddings",
        "body":{
            "model": embedding_model_name,
            "input": input
        }
    }
    
def save_results_as_jsonl(results, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for item in results:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")

def main(args):
    # --- Load dataset and generated answers ---
    with open(f"{args.dataset_dir}/{args.dataset_file}", "r", encoding="utf-8") as f:
        query_data = json.load(f)
    with open(f"{args.generation_dir}/{args.generation_env}_{args.generation_model_name.replace('/', '_')}.json", "r", encoding="utf-8") as f:
        answer_data = json.load(f)

    # --- Construct texts for embedding ---
    texts_answer_with_query = []
    answer_counts = []
    answer_indexes = []
    for task_index, (task, answers_per_task) in enumerate(zip(query_data, answer_data)):
        per_task_counts = []
        for query_index, (query, answers_per_query) in enumerate(zip(task["task_list"], answers_per_task)):
            per_task_counts.append(len(answers_per_query))
            for answer_index, answer in enumerate(answers_per_query):
                if args.embedding_only_answer:
                    texts_answer_with_query.append(answer)
                else:
                    texts_answer_with_query.append(f"Q: {task['task_prompt']} {query}\nA: {answer}")
                answer_indexes.append(f"{task_index}-{query_index}-{answer_index}")
        answer_counts.append(per_task_counts)

    batch_objects = [
        gpt_emb_batch(args.generation_env, index, answer, args.embedding_model_name)
        for index, answer in zip(answer_indexes, texts_answer_with_query)
    ]

    output_file_name = f"{args.batch_input_dir}/{args.generation_env}_{args.embedding_model_name}_{args.generation_model_name.replace('/', '_')}.jsonl"
    save_results_as_jsonl(batch_objects, output_file_name)

    print(f"Saved: {output_file_name}")

def upload_file_to_gpt(file_path):
    client = OpenAI()
    
    file = client.files.create(
        file = open(file_path, "rb"),
        purpose = "batch",
    )
    
    batch = client.batches.create(
        input_file_id = file.id,
        endpoint = "/v1/embeddings",
        completion_window = "24h"
    )
    
    return batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--generation_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--dataset_file", type=str, default="creative_task_sample.json")

    parser.add_argument("--generation_dir", type=str, default="generation")
    parser.add_argument("--generation_env", type=str, default="standard")
    parser.add_argument("--embedding_model_name", type=str, default="text-embedding-3-large")
    parser.add_argument("--batch_input_dir", type=str, default="openai_batch_input")
    parser.add_argument("--embedding_only_answer", action="store_true")

    parser.add_argument("--upload_to_gpt", action="store_true")
    args = parser.parse_args()

    main(args)

    if args.upload_to_gpt:
        output_file_name = f"{args.generation_env}_{args.embedding_model_name.split('/')[-1]}_{args.generation_model_name.replace('/', '_')}.jsonl"
        batch = upload_file_to_gpt(f"{args.batch_input_dir}/{output_file_name}")
        
        batch_info = {
            "batch_id": batch.id,
            "input_file_id": batch.input_file_id,
            "input_file_name": output_file_name
        }

        batch_info_file = f"{args.batch_input_dir}/batch_info.json"
        try:
            with open(batch_info_file, "r") as f:
                existing_info = json.load(f)
        except FileNotFoundError:
            existing_info = []
            
        existing_info.append(batch_info)
        
        with open(batch_info_file, "w") as f:
            json.dump(existing_info, f, indent=4)
        
        print(f"Uploaded to GPT: {output_file_name}")