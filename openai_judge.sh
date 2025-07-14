#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# ----------------------------------------------------------------------------
# Configuration of model names, environments, and directories
# ----------------------------------------------------------------------------
source "$(dirname "$0")/base.env"

mkdir -p "${BATCH_INPUT_DIR}" "${BATCH_OUTPUT_DIR}"

if $DO_UPLOAD; then
# ----------------------------------------------------------------------------
# Step 1: Upload batch input to OpenAI API (answers must be generated first)
# ----------------------------------------------------------------------------
    python b1_gpt_batch_upload.py \
        --dataset_dir "${DATASET_DIR}" \
        --dataset_file "${DATASET_FILE}" \
        --generation_model_name "${GEN_MODEL}" \
        --generation_dir "${GEN_DIR}" \
        --generation_env "${GEN_ENV}" \
        --judge_model_name "${JUDGE_MODEL}" \
        --batch_input_dir "${BATCH_INPUT_DIR}" \
        --upload_to_gpt
fi

if ! $DO_UPLOAD; then
# ----------------------------------------------------------------------------
# Step 2: Download batch output from OpenAI API and convert to CSV (if batch is completed)
# ----------------------------------------------------------------------------
    python b2_gpt_batch_download.py \
        --generation_model_name "${GEN_MODEL}" \
        --generation_env "${GEN_ENV}" \
        --judge_model_name "${JUDGE_MODEL}" \
        --batch_input_dir "${BATCH_INPUT_DIR}" \
        --batch_output_dir "${BATCH_OUTPUT_DIR}"

# ----------------------------------------------------------------------------
# Step 3: Calculate correlation between scores from OpenAI API and CAD scores
# ----------------------------------------------------------------------------
    python b3_corrlation_between_scores.py \
        --generation_model_name "${GEN_MODEL}" \
        --generation_env "${GEN_ENV}" \
        --embedding_model_name "${EMB_MODEL}" \
        --judge_model_name "${JUDGE_MODEL}" \
        --batch_output_dir "${BATCH_OUTPUT_DIR}" \
        --scoring_dir "${SCORING_DIR}"
fi