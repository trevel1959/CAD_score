import argparse
import json

import pandas as pd
import numpy as np

from openai import OpenAI

def make_batch_object(model, dataset_name, index, query = None, system_prompt = None):
    if system_prompt is None:
        system_prompt = (
            "You are an expert of psychology. Your objective is to assess the subject's creativity through their answers to some question/answering task related to divergent thinking. You will be given a question-answer pair. Your task is to score the answer."
            "\nYou should rate the answer on five metrics. For all five metrics, assign a score between 1 and 5, with 5 being the highest. Five metrics are:"
            "\n1) Fluency. Fluency refers to the ability to generate a large quantity of ideas or solutions to a given problem. This measure isn't concerned with the quality or uniqueness of the ideas, but rather the sheer volume. The more ideas one can produce, the higher the fluency is."
            "\n2) Flexibility. Flexibility is the capacity to shift one's thinking and to produce a wide range of ideas from different categories or perspectives. It involves being able to think outside of the box and to switch from one type of idea to another."
            "\n3) Originality. Originality refers to the ability to come up with unique or novel ideas that differ from the norm. It's not just about producing many ideas (fluency), but also about producing ideas that are different from what others might typically think of."
            "\n4) Elaboration. Elaboration is the ability to expand upon or add detail to ideas. It involves taking a simple idea and building upon it, adding complexity and depth. Elaboration isn't just about creating more, but about deepening what is there."
            "\n5) Finally, you will provide an overall score between 1 and 5, with 5 being the highest. You should only give the score, format like: Fluency: 3"
        )
    if model == "gpt-4-0613":   # only for gpt-4-0613
        return {
            "custom_id": f"{dataset_name}-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "creativity_score",
                            "description": "Assess the creativity of an answer based on five metrics.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "Fluency": {"type": "number"},
                                    "Flexibility": {"type": "number"},
                                    "Originality": {"type": "number"},
                                    "Elaboration": {"type": "number"},
                                    "Overall": {"type": "number"}
                                },
                                "required": ["Fluency", "Flexibility", "Originality", "Elaboration", "Overall"]
                            }
                        }
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {
                        "name": "creativity_score"
                    }
                }
            }
        }
    else:   # gpt-4o, gpt-4.1, and gpt-4o-mini, etc.
        return {
            "custom_id": f"{dataset_name}-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body":{
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "max_completion_tokens": 64,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "creativity_score",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "Fluency": {
                                    "type": "number"
                                },
                                "Flexibility": {
                                    "type": "number"
                                },
                                "Originality": {
                                    "type": "number"
                                },
                                "Elaboration": {
                                    "type": "number"
                                },
                                "Overall": {
                                    "type": "number"
                                }
                            },
                            "required": [
                                "Fluency",
                                "Flexibility",
                                "Originality",
                                "Elaboration",
                                "Overall"
                            ],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            }
        }

def upload_file_to_gpt(file_path):
    client = OpenAI()
    
    file = client.files.create(
        file = open(file_path, "rb"),
        purpose = "batch",
    )
    
    batch = client.batches.create(
        input_file_id = file.id,
        endpoint = "/v1/chat/completions",
        completion_window = "24h"
    )
    
    return batch

def main(args):
    with open(f"{args.dataset_dir}/{args.dataset_file}", "r") as f:
        query_list = json.load(f)
    with open(f"{args.generation_dir}/{args.generation_env}_{args.generation_model_name.replace('/', '_')}.json", "r") as f:
        answer_list = json.load(f)

    all_batch_objects = []
    for i, task_data in enumerate(query_list):
        for j, query in enumerate(task_data["task_list"]):
            query = f"Question: {task_data['task_prompt']}{query}\nAnswer: {' '.join(f'{j+1}. ' + answer for j, answer in enumerate(answer_list[i][j]))}"
            batch_object = make_batch_object(args.judge_model_name, task_data["task_name"], j, query)
            all_batch_objects.append(batch_object)

    output_file_dir = f"{args.batch_input_dir}/{args.generation_env}_{args.generation_model_name.replace('/', '_')}_{args.judge_model_name}.jsonl"
    with open(output_file_dir, "w") as f:
        for batch_object in all_batch_objects:
            f.write(json.dumps(batch_object) + "\n")
    
    print(f"Saved: {output_file_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--dataset_file", type=str, default="creative_task_sample.json")
    parser.add_argument("--generation_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--generation_dir", type=str, default="generation")
    parser.add_argument("--generation_env", type=str, default="standard")

    parser.add_argument("--judge_model_name", type=str, default="gpt-4.1")
    parser.add_argument("--batch_input_dir", type=str, default="openai_batch_input")

    parser.add_argument("--upload_to_gpt", action="store_true")
    args = parser.parse_args()

    main(args)

    if args.upload_to_gpt:
        batch = upload_file_to_gpt(f"{args.batch_input_dir}/{args.generation_env}_{args.generation_model_name.replace('/', '_')}_{args.judge_model_name}.jsonl")
        
        batch_info = {
            "batch_id": batch.id,
            "input_file_id": batch.input_file_id,
            "input_file_name": f"{args.generation_env}_{args.generation_model_name.replace('/', '_')}_{args.judge_model_name}.jsonl"
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