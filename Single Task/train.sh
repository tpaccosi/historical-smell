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
