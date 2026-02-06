#!/bin/bash

phase="train"
phase="hyperparam"
language="dutch"
phase=$1
language=$2
task_type=$3
fold=$4
invald=0
if [ "$language" == "" ];
then
	invalid=1
fi

if [ "$phase" != "hyperparam" ] && [ "$phase" != "train" ] && [ "$phase" != "test" ];
then
	invalid=1
fi

if [ "$task_type" == "single" ];
then
    data_dir="data/single"
    echo "task_type SINGLETASK, data_dir $data_dir"
elif [ "$task_type" == "multi" ];
then
    data_dir="data/multi"
    echo "task_type SINGLETASK, data_dir $data_dir"
else
	invalid=1
fi

if [ "$language" == "dutch" ];
then
  model="emanjavacas/GysBERT"
  learning_rate=3e-05
  train_batch_size=8
  train_epochs=14

elif [ "$language" == "english" ];
then
  model="emanjavacas/MacBERTh"
  learning_rate=5e-05
  train_batch_size=16
  train_epochs=8

elif [ "$language" == "french" ];
then
  model="pjox/dalembert"
  learning_rate=5e-05
  train_batch_size=32
  train_epochs=14

elif [ "$language" == "german_ghisbert" ];
then
  model="christinbeck/GHisBERT"
  language="german"
  learning_rate=3e-05
  train_batch_size=32
  train_epochs=8

elif [ "$language" == "german_bert" ];
then
  model="redewiedergabe/bert-base-historical-german-rw-cased"
  language="german"
  learning_rate=2e-05
  train_batch_size=8
  train_epochs=7

elif [ "$language" == "italian" ];
then
  model="bertoldo-all/checkpoint"
  learning_rate=5e-05
  train_batch_size=8
  train_epochs=16
else
	invalid=1
fi

if [ "$invalid" == 1 ];
then
	echo "	usage: $0 <phase> <language> <task_type> <fold>"
	echo "	phase must be 'hyperparam', 'test' or 'train'"
	echo "	language must be dutch, english, french, german_ghisbert german_bert or italian"
	echo "	task_type must single or multi"
	echo "	fold must between 1 and 5"
	exit
fi


echo "language: $language\nmodel: $model"

if [ "$phase" == "hyperparam" ];
then
  echo "Hyperparameter search."
  python3 train_single.py --hypsearch \
                                 --lang $language \
                                 --fold $fold \
                                 --data_dir $data_dir \
                                 --model $model
elif [ "$phase" == "train" ];
then
  echo "Train the model."
  python3 train_single.py --do_train --do_test \
                                 --lang $language \
                                 --fold $fold \
                                 --data_dir $data_dir \
                                 --learning_rate $learning_rate \
                                 --train_batch_size $train_batch_size\
                                 --train_epochs $train_epochs \
                                 --model $model

elif [ "$phase" == "test" ];
then
  echo "Test the model."
  python3 train_single.py --do_test \
                                 --lang $language \
                                 --fold $fold \
                                 --data_dir $data_dir \
                                 --model $model

fi                            
