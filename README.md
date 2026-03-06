# historical-smell

## Repository Overview

This repository presents the results of our experiments on **olfactory frame extraction**, comparing **contemporary** and **historical** language models. The goal of these experiments is to analyse how temporal changes in language affect frame extraction performance and to provide tools for reproducible evaluation.  

### In this repository, we provide:

- **Model Training and Prediction**  
  Scripts for training and running predictions in a **multitask setting** using the [**MAChamp**](https://github.com/machamp-nlp/machamp) toolkit.

- **Evaluation**
  Scripts to compute **precision, recall, and F1 scores**, supporting both **strict** and **lenient** evaluation modes, along with tools for more detailed **span-level analysis**.  

- **Subword Analysis**  
  Scripts to calculate **subword fertility rates** for each model and language, helping to understand if tokenisation affects model performance.  

- **Data Splits and Experiments**  
  Scripts to create **train/dev/test folds** and to generate **subsets of the training data** for experiments studying the **impact of training size**.

- **Span Detection Results**
  `partial_matches-start_end_early_late' shows the results of the span analysis in terms of start and end matches.

### Model Availability

The trained models will be released on **Hugging Face**. For **anonymity reasons during the peer-review period**, links to the models will only be provided after the anonymisation period ends.  
