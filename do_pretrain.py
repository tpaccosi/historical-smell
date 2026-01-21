from datasets import load_dataset

from pretraining.pretrain import prepare_model_and_tokenizer, prepare_dataset, make_trainer

def main():
    nl_texts = load_dataset(path='./data/pretrain/data_dutch/', data_files='texts_dutch.csv.gz', sep='\t', split='train')
    nl_texts = nl_texts.train_test_split(test_size=0.2)
    tokenizer, model = prepare_model_and_tokenizer('emanjavacas/GysBERT')
    lm_dataset, data_collator = prepare_dataset(tokenizer, nl_texts)
    trainer = make_trainer(model, lm_dataset, tokenizer, data_collator, output_dir='./models/pretrain/nl_model')

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
