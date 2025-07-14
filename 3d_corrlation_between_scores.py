import argparse
import pandas as pd
import json
import numpy as np
from scipy.stats import spearmanr, kendalltau
import pingouin as pg
from transformers import AutoTokenizer

def calculate_answer_lengths(generation_file, generation_model_name):
    tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
    
    # Calculate the length of each answer in the generation file
    with open(generation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    answer_lengths = []
    
    for i, task_answers in enumerate(data):
        for j, answer in enumerate(task_answers):
            length = np.mean([len(tokenizer.encode(ans)) for ans in answer])
            answer_lengths.append(length)
    
    return pd.DataFrame({
        'answer_length': answer_lengths
    })

def spearman_correlation(args):
    llm_result = pd.read_csv(f"{args.batch_output_dir}/_{args.generation_env}_{args.generation_model_name.replace('/', '_')}_{args.judge_model_name}.csv")
    emb_result = pd.read_csv(f"{args.scoring_dir}/{args.generation_env}_{args.embedding_model_name.split('/')[-1]}_{args.generation_model_name.replace('/', '_')}.csv")

    emb_result.columns = ["cad_score"]
    merged = pd.concat([llm_result, emb_result], axis=1)

    human_scores = ["Flexibility"]
    embed_scores = ["cad_score"]

    # Merge answer length information
    generation_file = f"generation/{args.generation_env}_{args.generation_model_name.replace('/', '_')}.json"
    length_df = calculate_answer_lengths(generation_file, args.generation_model_name)
    merged = pd.concat([merged, length_df], axis=1)

    print("=== Correlation Analysis ===")
    print(f"Generation Model: {args.generation_model_name}")
    print(f"Embedding Model: {args.embedding_model_name}")
    print(f"Judge Model: {args.judge_model_name}")
    print()

    for e in embed_scores:
        results = []

        for h in human_scores:
            # Zero-order Spearman correlation
            rho, pval_rho = spearmanr(merged[h], merged[e])
            
            # Partial correlation controlling for answer_length
            partial_corr = pg.partial_corr(
                data=merged, 
                x=h, 
                y=e, 
                covar='answer_length',
                method='spearman'
            )

            results.append({
                "score": h,
                "zero_order_rho": rho,
                "zero_order_p": pval_rho,
                "partial_rho": partial_corr['r'].iloc[0],
                "partial_p": partial_corr['p-val'].iloc[0],
                "n": len(merged)
            })

        df = pd.DataFrame(results)
        df.set_index("score", inplace=True)
        print(f"Embedding Score: {e}")
        print(df.round(4))
        print()
        
        # Analyze correlation change
        print("Correlation Change Analysis:")
        for _, row in df.iterrows():
            zero_order = row['zero_order_rho']
            partial = row['partial_rho']
            change = partial - zero_order
            change_percent = (change / abs(zero_order)) * 100 if zero_order != 0 else 0
            
            print(f"  {row.name}:")
            print(f"    Zero-order correlation: {zero_order:.4f} (p={row['zero_order_p']:.4f})")
            print(f"    Partial correlation: {partial:.4f} (p={row['partial_p']:.4f})")
            print(f"    Change: {change:.4f} ({change_percent:+.1f}%)")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--generation_env", type=str, default="standard_full")
    parser.add_argument("--embedding_model_name", type=str, default="Alibaba-NLP/gte-Qwen2-7B-instruct")
    parser.add_argument("--judge_model_name", type=str, default="gpt-4.1")
    parser.add_argument("--batch_output_dir", type=str, default="openai_output")
    parser.add_argument("--scoring_dir", type=str, default="scoring")
    args = parser.parse_args()

    spearman_correlation(args)