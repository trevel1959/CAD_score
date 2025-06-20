import argparse
import torch
import json
import re
import os
import logging
import time
import nltk
import string

from string import Template
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vllm import LLM, SamplingParams

def load_vllm_model(model_name: str) -> LLM:
    def make_llm(**extra_kwargs):
        return LLM(
            model=model_name,
            tokenizer=model_name,
            tokenizer_mode="auto",
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=True,
            dtype="half",
            **extra_kwargs
        )

    try:
        return make_llm()
    except ValueError as e:
        msg = str(e)

        if m := re.search(r"KV cache \((\d+)\)", msg):
            cap = int(m.group(1))
            return make_llm(max_model_len=cap)
        else:
            raise

def extract_items(text: str) -> list[str]:
    if text is None or text == "":
        return []

    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r'-{3,}', ' ', text)
    text = re.sub(r'\n(\d+[.:])\s+', r'<<SPLIT>>\1', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\*+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    text = text.replace('<<SPLIT>>', '\n')

    lines = text.splitlines()

    numbered_items = []

    for line in lines:
        line = line.strip()
        match = re.match(r'.*?\b\d+[.:]\s*(.*)', line)
        if match:
            parsed_item = match.group(1).strip()
            if parsed_item:
                numbered_items.append(parsed_item)

    return numbered_items

def is_answer_valid(ans: str, finish_reason: str, args: argparse.Namespace) -> bool:
    # 1. extract items
    try:
        items = extract_items(ans)
        
        if finish_reason != "stop":
            items = items[:-1]
    except Exception as e:
        return False

    # 2. check required number of items
    if args.minimum_required_num_items is not None:
        if len(items) < args.minimum_required_num_items:
            return False
            
    if args.maximum_allowed_num_items is not None:
        items = items[:args.maximum_allowed_num_items]

    # 3. remove stopwords and punctuation, and check if the answer is meaningful
    if args.language == "english":
        stop_words = set(stopwords.words("english"))
        punctuation_table = str.maketrans('', '', string.punctuation)

        def is_meaningful_token(item: str) -> bool:
            meaningful_tokens = [
                w
                for w in word_tokenize(item.lower().translate(punctuation_table))
                if w.isalpha() and w not in stop_words
            ]
            return len(meaningful_tokens) >= args.min_meaningful_tokens_per_answer

    # elif args.language == "chinese":
    #     import jieba

    #     stop_words = set(stopwords.words("chinese"))
    #     punctuation_table = str.maketrans('', '', string.punctuation)

    #     def is_meaningful_token(item: str) -> bool:
    #         tokens = jieba.lcut(item.translate(punctuation_table))
    #         meaningful_tokens = [
    #             w for w in tokens if w.strip() and w not in stop_words
    #         ]
    #         return len(meaningful_tokens) >= args.min_meaningful_tokens_per_answer
    else:
        raise ValueError(f"Invalid language: {args.language}")

    # 4. check if the answer is meaningful
    return all(is_meaningful_token(item) for item in items)

def init_debug_log(output_dir: str, model_name: str, n_queries: int, time_str: str) -> tuple[str, dict]:
    os.makedirs("./debug", exist_ok=True)
    debug_file = f"./debug/{time_str}_{output_dir}_{model_name.replace('/', '_')}.json"

    if os.path.exists(debug_file):
        with open(debug_file, "r", encoding="utf-8") as f:
            debug_log = json.load(f)
    else:
        debug_log = {"success": [None] * n_queries, "fail": []}

    return debug_file, debug_log

def save_debug_log(debug_file: str, debug_log: dict, success_entries: list[tuple[int, str]], fail_entries: list[tuple[int, str]]):
    for idx, ans in success_entries:
        parsed = extract_items(ans)
        debug_log["success"][idx] = {
            "raw_answer": ans,
            "parsed_answer": parsed
        }

    if fail_entries:
        fail_log = []
        for idx, ans in fail_entries:
            parsed = extract_items(ans)
            fail_log.append({
                "index": idx,
                "raw_answer": ans,
                "parsed_answer": parsed
            })

        debug_log["fail"].append({
            "generated_time": time.time(),
            "fail_query_list": fail_log
        })

    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(debug_log, f, ensure_ascii=False, indent=2)

def generate_with_validation(queries: list[str], tokenizer, args: argparse.Namespace, max_retries: int = 20) -> list[list[str]]:
    # --- Initialization ---
    start_time = time.time()
    time_str = time.strftime("%m%d_%H%M%S", time.localtime(start_time))

    n_queries = len(queries)
    answers = [None] * n_queries
    invalid_indices = list(range(n_queries))

    if args.temp_save:
        debug_file, debug_log = init_debug_log(args.generation_dir, args.generation_model_name, n_queries, time_str)

    model = load_vllm_model(args.generation_model_name)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed = 42,
    )

    if args.top_p is not None:
        sampling_params.top_p = args.top_p
    if args.top_k is not None:
        sampling_params.top_k = args.top_k
    if args.min_p is not None:
        sampling_params.min_p = args.min_p

    # --- Retry Loop ---
    retries = 0
    while invalid_indices and retries < max_retries:
        batch_queries = [queries[i] for i in invalid_indices]

        outputs = model.generate(batch_queries, sampling_params=sampling_params, use_tqdm=True)
        re_answers = [o.outputs[0].text.strip() for o in outputs]
        finish_reasons = [o.outputs[0].finish_reason for o in outputs]

        success_entries, fail_entries = [], []
        for idx, new_ans, finish_reason in zip(invalid_indices, re_answers, finish_reasons):
            if is_answer_valid(
                new_ans,
                finish_reason,
                args
            ):
                answers[idx] = new_ans
                success_entries.append((idx, new_ans))
            else:
                fail_entries.append((idx, new_ans))

        if args.temp_save:
            save_debug_log(debug_file, debug_log, success_entries, fail_entries)

        invalid_indices = [idx for idx, _ in fail_entries]
        retries += 1
        sampling_params.seed += 1

    if args.maximum_allowed_num_items is not None:
        parsed_answers = [extract_items(ans)[:args.maximum_allowed_num_items] for ans in answers]
    else:
        parsed_answers = [extract_items(ans) for ans in answers]

    return parsed_answers

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.generation_model_name, trust_remote_code=True, use_fast=True)

    # --- Construct queries from dataset ---
    with open(f"{args.dataset_dir}/{args.dataset_file}", 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if args.minimum_required_num_items is not None:
        items_str = f"{args.minimum_required_num_items} "
    else:
        items_str = ""

    system_prompt = (
        f"\nFor each of the following questions, generate {items_str}creative and original ideas. "
        "Provide an explanation for each idea. "
        "List the ideas using numbered bullets (e.g., \"1.\", \"2.\", etc.)."
        "Use a line break only to separate each idea. "
        "Do not include any introductory text."
    )

    use_system_role = "gemma" not in args.generation_model_name.lower()

    all_queries = []
    for task in dataset:
        task_prompt = task["task_prompt"]
        for task_query in task["task_list"]:
            if use_system_role:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_prompt + task_query}
                ]
            else:
                messages = [{"role": "user", "content": system_prompt + "\n" + task_prompt + task_query}]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            all_queries.append(prompt)

    # --- Generate and save results ---
    all_answers = generate_with_validation(all_queries, tokenizer, args)

    final_answers = []
    idx = 0

    for task in dataset:
        task_queries = []
        for _ in task["task_list"]:
            task_queries.append(all_answers[idx])
            idx += 1
        final_answers.append(task_queries)

    gen_file_path = f"{args.generation_dir}/{args.generation_env}_{args.generation_model_name.replace('/', '_')}.json"
    with open(gen_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_answers, f, ensure_ascii=False, indent=4)

    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def convert_or_none(type_fn):
    def convert(x):
        if x.lower() == "none":
            return None
        return type_fn(x)
    return convert

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--generation_model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--dataset_file", type=str, default="creative_task_sample.json")
    parser.add_argument("--language", type=str, default="english")

    parser.add_argument("--generation_dir", type=str, default="generation")
    parser.add_argument("--generation_env", type=str, default="standard")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=convert_or_none(float), default=1)
    parser.add_argument("--min_p", type=convert_or_none(float), default=None)
    parser.add_argument("--top_k", type=convert_or_none(int), default=50)
    parser.add_argument("--minimum_required_num_items", type=convert_or_none(int), default=None)
    parser.add_argument("--maximum_allowed_num_items", type=convert_or_none(int), default=None)
    parser.add_argument("--min_meaningful_tokens_per_answer", type=convert_or_none(int), default=3)

    parser.add_argument("--temp-save", action="store_true")
    args = parser.parse_args()

    logging.getLogger("vllm").setLevel(logging.ERROR)
    main(args)