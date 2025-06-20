import argparse
import pandas as pd
from scipy.stats import spearmanr, kendalltau

def spearman_correlation(args):
    llm_result = pd.read_csv(f"{args.batch_output_dir}/{args.generation_env}_{args.generation_model_name.replace('/', '_')}_{args.judge_model_name}.csv")
    emb_result = pd.read_csv(f"{args.scoring_dir}/{args.generation_env}_{args.embedding_model_name.split('/')[-1]}_{args.generation_model_name.replace('/', '_')}.csv")

    emb_result.columns = ["cad_score"]
    merged = pd.concat([llm_result, emb_result], axis=1)

    # human_scores = ["Fluency", "Flexibility", "Originality", "Elaboration", "Overall"]
    human_scores = ["Flexibility"]
    embed_scores = ["cad_score"]

    merged['task_prefix'] = merged['custom_id'].str.split('-').str[0]

    for e in embed_scores:
        results = []

        for h in human_scores:
            # Spearman
            rho, pval_rho = spearmanr(merged[h], merged[e])

            results.append({
                "score": h,
                "spearman rho": rho,
                "p-value": pval_rho
            })

        df = pd.DataFrame(results)
        df.set_index("score", inplace=True)
        print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--generation_env", type=str, default="standard")
    parser.add_argument("--embedding_model_name", type=str, default="Alibaba-NLP/gte-Qwen2-7B-instruct")
    parser.add_argument("--judge_model_name", type=str, default="gpt-4.1")
    parser.add_argument("--batch_output_dir", type=str, default="openai_output")
    parser.add_argument("--scoring_dir", type=str, default="scoring")
    args = parser.parse_args()

    spearman_correlation(args)