import json
import argparse
import pickle
import zstandard as zstd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_embeddings_realtime(args):
    # --- 데이터 로드 ---
    with open(f"{args.dataset_dir}/{args.dataset_file}", "r", encoding="utf-8") as f:
        query_data = json.load(f)
    with open(f"{args.generation_dir}/{args.generation_env}_{args.generation_model_name.replace('/', '_')}.json", "r", encoding="utf-8") as f:
        answer_data = json.load(f)

    # --- 임베딩 입력 텍스트 생성 ---
    texts_answer_with_query = []
    answer_indexes = []
    for task_index, (task, answers_per_task) in enumerate(zip(query_data, answer_data)):
        for query_index, (query, answers_per_query) in enumerate(zip(task["task_list"], answers_per_task)):
            for answer_index, answer in enumerate(answers_per_query):
                if args.embedding_only_answer:
                    texts_answer_with_query.append(answer)
                else:
                    texts_answer_with_query.append(f"Q: {task['task_prompt']} {query}\nA: {answer}")
                answer_indexes.append(f"{task_index}-{query_index}-{answer_index}")

    # --- OpenAI 임베딩 API 병렬 호출 ---
    client = OpenAI()
    embeddings = [None] * len(texts_answer_with_query)
    max_workers = 10  # 상황에 맞게 조절 (rate limit에 따라 조정)

    def fetch_embedding(text):
        try:
            response = client.embeddings.create(
                model=args.embedding_model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_embedding, text): i
            for i, text in enumerate(texts_answer_with_query)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding (realtime-parallel)"):
            idx = futures[future]
            embeddings[idx] = future.result()

    # --- 인덱스별로 nested 구조로 변환 ---
    from collections import defaultdict
    nested = defaultdict(lambda: defaultdict(list))
    for idx, emb in zip(answer_indexes, embeddings):
        t_str, q_str, a_str = idx.split("-")
        t, q = int(t_str), int(q_str)
        nested[t][q].append(emb)
    grouped = [
        [nested[t][q] for q in sorted(nested[t].keys())]
        for t in sorted(nested.keys())
    ]

    # --- 저장 ---
    output_file = (
        f"{args.embedding_dir}/{args.generation_env}_{args.embedding_model_name}_{args.generation_model_name.replace('/', '_')}.zstd"
        if not args.embedding_only_answer else
        f"{args.embedding_dir}/{args.generation_env}_{args.embedding_model_name}_{args.generation_model_name.replace('/', '_')}_answer_only.zstd"
    )
    os.makedirs(args.embedding_dir, exist_ok=True)
    with open(output_file, "wb") as f:
        cctx = zstd.ZstdCompressor(level=5)
        with cctx.stream_writer(f) as writer:
            pickle.dump(grouped, writer, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved embeddings to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--dataset_file", type=str, default="creative_task.json")

    parser.add_argument("--generation_dir", type=str, default="generation")
    parser.add_argument("--generation_env", type=str, default="standard")

    parser.add_argument("--embedding_model_name", type=str, default="text-embedding-3-large")
    parser.add_argument("--embedding_dir", type=str, default="embedding")
    parser.add_argument("--embedding_only_answer", action="store_true")
    args = parser.parse_args()
    get_embeddings_realtime(args) 