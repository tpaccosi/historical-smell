#!/bin/bash

todo="train"
todo="hyperparam"
language="dutch"
todo=$1
language=$2
task_type=$3

if [ "$language" == "" ];
then
	echo "second argument must be dutch, english, french, german_ghisbert german_bert or italian"
	exit
fi

if [ "$todo" != "hyperparam" ] && [ "$todo" != "train" ];
then
	echo "first argument must be 'hyperparam' or 'train'"
	exit
fi

if [ "$task_type" == "single" ];
then
    data_dir="../data/single"
    echo "task_type SINGLETASK, data_dir $data_dir"
elif [ "$task_type" == "multi" ];
then
    data_dir="../data/multi"
    echo "task_type SINGLETASK, data_dir $data_dir"
else
    echo "Must provide a task_type (single or multi)"
    echo "Usage: train.sh <phase> <language> <task_type>"
    echo "    where phase is one of 'hyperparam', 'train' or 'test'"
    exit
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
fi

echo "language: $language\nmodel: $model"

if [ "$todo" == "hyperparam" ];
then
  echo "Hyperparameter search."
  python3 train.py --hypsearch \
                                 --lang $language \
                                 --fold 1 \
                                 --data_dir $data_dir \
                                 --model $model
elif [ "$todo" == "train" ];
then
  echo "Train the model."
  python3 train.py --do_train --do_test \
                                 --lang $language \
                                 --fold 1 \
                                 --data_dir $data_dir \
                                 --learning_rate $learning_rate \
                                 --train_batch_size $train_batch_size\
                                 --train_epochs $train_epochs \
                                 --model $model

elif [ "$todo" == "test" ];
then
  echo "Test the model."
  python3 train.py --do_test \
                                 --lang $language \
                                 --fold 1 \
                                 --data_dir $data_dir \
                                 --model $model

fi                            
