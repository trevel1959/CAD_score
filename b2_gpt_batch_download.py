import argparse
import json
import pandas as pd
from openai import OpenAI

def download_and_convert_batch_result(args):
    # --- Find batch ID ---
    with open(f"{args.batch_input_dir}/batch_info.json", "r") as f:
        batches = json.load(f)

    input_file_name = f"{args.generation_env}_{args.generation_model_name.replace('/', '_')}_{args.judge_model_name}.jsonl"

    batch_id = None
    for batch in batches:
        if batch["input_file_name"] == input_file_name:
            batch_id = batch["batch_id"]
            break
    if batch_id is None:
        raise ValueError(f"Batch ID not found for input file: {input_file_name}")

    # --- Download and process batch result ---
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        raise ValueError(f"Batch {batch_id} is not completed. Status: {batch.status}")

    batch_content = client.files.content(batch.output_file_id).text

    rows = []
    for line in batch_content.strip().split('\n'):
        data = json.loads(line)
        if "tool_calls" in data["response"]["body"]["choices"][0]["message"]:  # gpt-4-0613
            content_str = data["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        else:  # gpt-4o, gpt-4.1 etc.
            content_str = data["response"]["body"]["choices"][0]["message"]["content"]
        content_dict = json.loads(content_str)
        content_dict["custom_id"] = data.get("custom_id", "")
        rows.append(content_dict)

    df = pd.DataFrame(rows)
    csv_output_file_name = f"{args.generation_env}_{args.generation_model_name.replace('/', '_')}_{args.judge_model_name}.csv"
    csv_output_path = f"{args.batch_output_dir}/{csv_output_file_name}"
    df.to_csv(csv_output_path, index=False)

    print(f"Downloaded batch result and saved to CSV: {csv_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--generation_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--generation_env", type=str, default="standard")
    parser.add_argument("--batch_input_dir", type=str, default="batch_input")
    parser.add_argument("--batch_output_dir", type=str, default="batch_output")
    args = parser.parse_args()

    download_and_convert_batch_result(args)