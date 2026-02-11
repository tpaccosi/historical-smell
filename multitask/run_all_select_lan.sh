#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash $0 {en|de|fr|nl|it}"
    exit 1
fi

LANG=$1

if [[ ! "$LANG" =~ ^(en|de|fr|nl|it)$ ]]; then
    echo "Invalid language. Choose: en, de, fr, nl, it"
    exit 1
fi


OUTPUT_FILE="frac_0.8-results-${LANG}-span-smell.txt"
MODEL_NAME="smell-model-${LANG}-30-frac_0.8"
MAKE_JSON_SCRIPT="make_json/make-json-${LANG}.py"
PARAMS_CONFIG="configs/params-${LANG}.json"
PREDICTIONS_PATH="predictions-mt/predictions-${LANG}-frac_0.8/"
DATA_FOLDER="mt-all-lang/out-fold-smell-${LANG}/"

FOLDS=(0 1 2 3 4)

WEIGHTS=(1 1 1 1 1 1 1 1 1 1)

> "${OUTPUT_FILE}"

for i in "${FOLDS[@]}"; do
    echo "Fold ${i} (${LANG})"

    TRAIN_FILE="${DATA_FOLDER}folds_${i}_train-train_size_0.8.tsv"
    DEV_FILE="${DATA_FOLDER}folds_${i}_dev.tsv"
    TEST_FILE="${DATA_FOLDER}folds_${i}_test.tsv"

    JSON_CONFIG="smell_seq-tasks-${LANG}.json"

    
    python3 "${MAKE_JSON_SCRIPT}" \
        "${TRAIN_FILE}" \
        "${DEV_FILE}" \
        "${WEIGHTS[@]}" \
        > "${JSON_CONFIG}"

    
    python3 train.py \
        --dataset_configs "${JSON_CONFIG}" \
        --parameters_config "${PARAMS_CONFIG}" \
        --device 0 \
        --name "${MODEL_NAME}"

    
    LAST_PATH=$(ls -td logs/${MODEL_NAME}/*/ | head -1)

    METRICS_PATH="${LAST_PATH}metrics.json"
    MODEL_PATH="${LAST_PATH}model.pt"

    {
        echo "Fold: ${i}"
        echo "Language: ${LANG}"
        echo "Run path: ${LAST_PATH}"
        echo "Weights: ${WEIGHTS[*]}"
        echo "Dev results (best):"
        grep best "${METRICS_PATH}" || echo "No 'best' entry found"
        echo ""
    } >> "${OUTPUT_FILE}"

    
    PRED_FILE="${PREDICTIONS_PATH}${i}_prediction-frac_0.8.tsv"

    python3 predict.py \
        "${MODEL_PATH}" \
        "${TEST_FILE}" \
        "${PRED_FILE}" \
        --device 0

    echo "Test results:" >> "${OUTPUT_FILE}"
    if [ -f "${PRED_FILE}.eval" ]; then
        cat "${PRED_FILE}.eval" >> "${OUTPUT_FILE}"
    else
        echo "Eval file not found for fold ${i}" >> "${OUTPUT_FILE}"
    fi
    echo "" >> "${OUTPUT_FILE}"

done
