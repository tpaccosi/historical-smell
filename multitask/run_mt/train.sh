#!/bin/bash

OUTPUT_FILE="30-ep-results-de-span-smell.txt"

MODEL_NAME="smell-model-de-30"

MAKE_JSON_PATH=""
PREDICTIONS_PATH="predictions/predictions-de/"

dataFolder="out-fold-smell-de/" 

folds=(0 1 2 3 4)

weight1=1
weight2=1
weight3=1
weight4=1
weight5=1
weight6=1
weight7=1
weight8=1
weight9=1
weight10=1

for i in "${folds[@]}"
	do	
			python3 ${MAKE_JSON_PATH}make-json-de.py ${dataFolder}folds_${i}_train.tsv ${dataFolder}folds_${i}_dev.tsv $weight1 $weight2 $weight3 $weight4 $weight5 $weight6 $weight7 $weight8 $weight9 $weight10 > smell_seq-tasks-de.json

			python3 train.py --dataset_configs smell_seq-tasks-de.json --parameters_config configs/params-de.json --device 0 --name ${MODEL_NAME}

			LAST_PATH=$(ls -td logs/$MODEL_NAME/*/ | head -1)

			DEV_RESULTS_PATH=${LAST_PATH}metrics.json

			configString="Fold: ${i} ${MODEL_NAME} weights: $weight1 $weight2 $weight3 $weight4 $weight5 $weight6 $weight7 $weight8 $weight9 $weight10"
			
			echo $DEV_RESULTS_PATH >> ${OUTPUT_FILE}
			echo $configString >> ${OUTPUT_FILE}
			echo "dev results" >>  ${OUTPUT_FILE}
			cat $DEV_RESULTS_PATH | grep best >> ${OUTPUT_FILE}

			MODEL_PATH=${LAST_PATH}model.pt

	done