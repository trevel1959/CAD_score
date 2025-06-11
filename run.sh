#!/bin/bash

# Download nltk resources
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
"

# Define an array of models
models=(
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"

    "google/gemma-2-2b-it"
    "google/gemma-2-9b-it"

    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
)

# Make required directories
debug_dir="debug"
generation_dir="generation"
embedding_dir="embedding"
scoring_dir="scoring"

mkdir -p "${generation_dir}" "${embedding_dir}" "${debug_dir}" "${scoring_dir}"

for model in "${models[@]}"; do
    generation_env_name="standard"
    embedding_env_name="gte-Qwen2-7B-instruct"

    echo "[Generating] Model: $model"
    python a1_inference.py \
    -m "${model}" \
    --dataset_file "creative_task_sample.json" \
    --generation_env "${generation_env_name}" \
    --temperature 1 --top_p 1 --top_k 50 --min_p None \
    --minimum_required_num_items 5 --maximum_allowed_num_items 5 \
    --temp-save

    echo "[Embedding] Model: $model"
    python a2_embedding.py \
    -m "${model}" \
    --embedding_model_name "Alibaba-NLP/gte-Qwen2-7B-instruct" \
    --generation_env "${generation_env_name}" \
    --embedding_env "${generation_env_name}_${embedding_env_name}"

    echo "[Scoring] Model: $model"
    python a3_scoring.py \
    -m "${model}" \
    --embedding_dir "${embedding_dir}" \
    --embedding_env "${generation_env_name}_${embedding_env_name}" \
    --scoring_dir "${scoring_dir}" \
    --scoring_env "${generation_env_name}_${embedding_env_name}"
done