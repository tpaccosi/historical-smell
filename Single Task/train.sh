#!/bin/bash

todo="train"
todo="hyperparam"
language="dutch"

if [ "$language" == "dutch" ];
then
  model="emanjavacas/GysBERT"

elif [ $language == "english" ];
then
  model="emanjavacas/MacBERTh"

elif [ $language == "french" ];
then
  model="pjox/dalembert"

elif [ $language == "german_ghisbert" ];
then
  model="christinbeck/GHisBERT"
  language = "german"

elif [ $language == "german_bert" ];
then
  model="bert-base-historical-german-rw-cased"
  language = "german"

elif [ $language == "italian" ];
then
  model="bertoldo-all/checkpoint"
fi

if [ "$todo" == "hyperparam" ];
then
  echo "Hyperparameter search."
  python3 train.py --hypsearch \
                                 --lang $language \
                                 --fold 1 \
                                 --model $model
elif [ "$todo" == "train" ];
then
  echo "Train the model."
  python3 train.py --do_train --do_test \
                                 --lang $language \
                                 --fold 1 \
                                 --learning_rate 5e-05 \
                                 --train_batch_size 16\
                                 --train_epochs 8 \
                                 --model $model

elif [ "$todo" == "test" ];
then
  echo "Test the model."
  python3 train.py --do_test \
                                 --lang $language \
                                 --fold 1 \
                                 --model $model

fi                            
