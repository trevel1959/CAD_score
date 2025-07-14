#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

source "$(dirname "$0")/base.env"

EMB_MODEL_TAG="${EMB_MODEL}"
if [ "$SCORING_METHOD" != "None" ]; then
    echo "[Scoring] Model: $GEN_MODEL"
    python 3c_scoring_other_methods.py \
        --generation_model_name "$GEN_MODEL" \
        --generation_dir "$GEN_DIR" \
        --generation_env "$GEN_ENV" \
        --scoring_dir "$SCORING_DIR" \
        --scoring_method "$SCORING_METHOD" \
        --arg_n "$SCORING_ARG_N"

    EMB_MODEL_TAG="_${SCORING_METHOD}"
fi

python 3d_corrlation_between_scores.py \
    --generation_model_name "$GEN_MODEL" \
    --generation_env "$GEN_ENV" \
    --embedding_model_name "$EMB_MODEL_TAG" \
    --judge_model_name "$JUDGE_MODEL" \
    --batch_output_dir "$BATCH_OUTPUT_DIR" \
    --scoring_dir "$SCORING_DIR"