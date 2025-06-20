#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

source "$(dirname "$0")/.env"

if $DO_UPLOAD; then
# ----------------------------------------------------------------------------
# Step 1: Create batch input for generated answers and upload to OpenAI API (answers must be generated first)
# ----------------------------------------------------------------------------
    echo "[Embedding] Model: $GEN_MODEL"
    python c1_gpt_emb_batch_upload.py \
        --dataset_dir "$DATASET_DIR" \
        --dataset_file "$DATASET_FILE" \
        --generation_model_name "$GEN_MODEL" \
        --generation_dir "$GEN_DIR" \
        --generation_env "$GEN_ENV" \
        --embedding_model_name "$API_EMB_MODEL" \
        --batch_input_dir "$BATCH_INPUT_DIR" \
        --upload_to_gpt
fi

if ! $DO_UPLOAD; then
# ----------------------------------------------------------------------------
# Step 2: Download embeddings from OpenAI API and save as zstd file (if batch is completed)
# ----------------------------------------------------------------------------
    python c2_gpt_emb_batch_download.py \
        --generation_model_name "$GEN_MODEL" \
        --generation_env "$GEN_ENV" \
        --embedding_model_name "$API_EMB_MODEL" \
        --embedding_dir "$EMB_DIR" \
        --batch_input_dir "$BATCH_INPUT_DIR"

    python a3_scoring.py \
        --generation_model_name "$GEN_MODEL" \
        --generation_env "$GEN_ENV" \
        --embedding_model_name "$API_EMB_MODEL" \
        --embedding_dir "$EMB_DIR" \
        --scoring_dir "$SCORING_DIR"
fi