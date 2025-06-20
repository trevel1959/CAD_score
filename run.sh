#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# Download required NLTK resources for text processing
python - <<PYCODE
import nltk

# Download tokenization and stopword data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
PYCODE

# ----------------------------------------------------------------------------
# Source shared configuration
# ----------------------------------------------------------------------------
source "$(dirname "$0")/.env"

# Create output directories if they do not already exist
mkdir -p "$GEN_DIR" "$EMB_DIR" "$DEBUG_DIR" "$SCORING_DIR"

# ----------------------------------------------------------------------------
# Step 1: Generate creative answers
# ----------------------------------------------------------------------------
echo "[Generating] Model: $GEN_MODEL"
python a1_inference.py \
    --dataset_dir "$DATASET_DIR" \
    --dataset_file "$DATASET_FILE" \
    --generation_model_name "$GEN_MODEL" \
    --generation_dir "$GEN_DIR" \
    --generation_env "$GEN_ENV" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --min_p "$MIN_P" \
    --minimum_required_num_items "$MIN_REQUIRED_NUM_ITEMS" \
    --maximum_allowed_num_items "$MAX_ALLOWED_NUM_ITEMS" \
    --min_meaningful_tokens_per_answer "$MIN_MEANINGFUL_TOKENS_PER_ANSWER"

# ----------------------------------------------------------------------------
# Step 2: Compute embeddings for generated answers
# ----------------------------------------------------------------------------
echo "[Embedding] Model: $GEN_MODEL"
python a2_embedding.py \
    --dataset_dir "$DATASET_DIR" \
    --dataset_file "$DATASET_FILE" \
    --generation_model_name "$GEN_MODEL" \
    --generation_dir "$GEN_DIR" \
    --generation_env "$GEN_ENV" \
    --embedding_model_name "$EMB_MODEL" \
    --embedding_dir "$EMB_DIR"

# ----------------------------------------------------------------------------
# Step 3: Calculate CAD scores using embeddings
# ----------------------------------------------------------------------------
echo "[Scoring] Model: $GEN_MODEL"
python a3_scoring.py \
    --generation_model_name "$GEN_MODEL" \
    --generation_env "$GEN_ENV" \
    --embedding_model_name "$EMB_MODEL" \
    --embedding_dir "$EMB_DIR" \
    --scoring_dir "$SCORING_DIR"

echo "All steps completed successfully."
