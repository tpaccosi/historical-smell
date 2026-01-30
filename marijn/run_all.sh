#!/bin/bash

OUTPUT_FILE="30-ep-results-en-span-smell.txt"

MODEL_NAME="smell-model-30"

MAKE_JSON_PATH=""
PREDICTIONS_PATH="predictions/"

dataFolder="out-fold-smell-en/" 

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
			python3 ${MAKE_JSON_PATH}make-json.py ${dataFolder}folds_${i}_train.tsv ${dataFolder}folds_${i}_dev.tsv $weight1 $weight2 $weight3 $weight4 $weight5 $weight6 $weight7 $weight8 $weight9 $weight10 > smell_seq-tasks.json

			python3 train.py --dataset_configs smell_seq-tasks.json --parameters_config configs/params-en.json --device 0 --name ${MODEL_NAME}

			LAST_PATH=$(ls -td logs/$MODEL_NAME/*/ | head -1)

			DEV_RESULTS_PATH=${LAST_PATH}metrics.json

			configString="Fold: ${i} ${MODEL_NAME} weights: $weight1 $weight2 $weight3 $weight4 $weight5 $weight6 $weight7 $weight8 $weight9 $weight10"
			
			echo $DEV_RESULTS_PATH >> ${OUTPUT_FILE}
			echo $configString >> ${OUTPUT_FILE}
			echo "dev results" >>  ${OUTPUT_FILE}
			cat $DEV_RESULTS_PATH | grep best >> ${OUTPUT_FILE}

			MODEL_PATH=${LAST_PATH}model.pt

			python3 predict.py ${MODEL_PATH} ${dataFolder}folds_${i}_test.tsv ${PREDICTIONS_PATH}${i}_${weight1}_${weight2}_${weight3}_${weight4}_${weight5}_${weight6}_${weight7}_${weight8}_${weight9}_${weight10}_prediction.tsv  --device 0

			echo "test results" >>  ${OUTPUT_FILE}
			cat ${PREDICTIONS_PATH}${i}_${weight1}_${weight2}_${weight3}_${weight4}_${weight5}_${weight6}_${weight7}_${weight8}_${weight9}_${weight10}_prediction.tsv.eval >> ${OUTPUT_FILE}			
			echo "" >> ${OUTPUT_FILE}

			# rm ${MODEL_PATH}
			# W_PATH=${LAST_PATH}weights.th
			# rm ${W_PATH}
					
	done