#!/bin/bash

todo="train"
# todo="hyperparam"

if [ "$todo" == "hyperparam" ];
then
  echo "Hyperparameter search."
  python3 train.py --hypsearch \
                                 --lang "english" \
                                 --fold 1 \
                                 --model "emanjavacas/MacBERTh"
elif [ "$todo" == "train" ];
then
  echo "Train the model."
  python3 train.py --do_train --do_test \
                                 --lang "english" \
                                 --fold 1 \
                                 --learning_rate 5e-05 \
                                 --train_batch_size 16\
                                 --train_epochs 8 \
                                 --model "emanjavacas/MacBERTh"

elif [ "$todo" == "test" ];
then
  echo "Test the model."
  python3 train.py --do_test \
                                 --lang "english" \
                                 --fold 1 \
                                 --model "emanjavacas/MacBERTh"

fi                            
