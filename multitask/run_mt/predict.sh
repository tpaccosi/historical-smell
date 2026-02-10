#!/bin/bash

OUTPUT_FILE="all_results-nl-span-smell.txt"
PREDICTIONS_PATH="predictions/predictions-nl/"
dataFolder="out-fold-smell-nl/" 

folds=(0 1 2 3 4)

# Assuming saved in logs/smell-model/fold_i/model.pt
MODEL_BASE_PATH="logs/smell-model-nl-30"

> ${OUTPUT_FILE}

for i in "${folds[@]}"
do
    MODEL_PATH="${MODEL_BASE_PATH}/fold_${i}/model.pt"
    TEST_FILE="${dataFolder}folds_${i}_test.tsv"
    PRED_FILE="${PREDICTIONS_PATH}${i}_prediction.tsv"

    echo "Predicting fold $i using model $MODEL_PATH"

    python3 predict.py ${MODEL_PATH} ${TEST_FILE} ${PRED_FILE} --device 0

    echo "Fold: $i" >> ${OUTPUT_FILE}
    echo "Model: ${MODEL_PATH}" >> ${OUTPUT_FILE}
    echo "Test results:" >> ${OUTPUT_FILE}
    if [ -f "${PRED_FILE}.eval" ]; then
        cat ${PRED_FILE}.eval >> ${OUTPUT_FILE}
    else
        echo "Eval file not found for fold $i" >> ${OUTPUT_FILE}
    fi
    echo "" >> ${OUTPUT_FILE}
done
