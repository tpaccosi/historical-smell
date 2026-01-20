from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import default_data_collator
from transformers import TrainingArguments
from transformers import Trainer


BLOCK_SIZE = 128


def make_trainer(model, lm_dataset, tokenizer, data_collator, output_dir: str):
    batch_size = 64
    # Show the training loss with every epoch
    logging_steps = len(lm_dataset["train"]) // batch_size
    model_name = 'GysBERT'

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        push_to_hub=False,
        fp16=False,  # doesn't work with mps / Apple M1?
        logging_steps=logging_steps,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )


def prepare_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return tokenizer, model


def prepare_dataset(tokenizer, texts):
    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["text"]])

    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    tokenized_texts = texts.map(
        tokenize_function,
        batched=True,
        num_proc=6,
        remove_columns=texts["train"].column_names,
    )

    lm_dataset = tokenized_texts.map(group_texts, batched=True, num_proc=6)
    tokenizer.eos_token

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    return lm_dataset, data_collator


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= BLOCK_SIZE:
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    # Split by chunks of BLOCK_SIZE.
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    return result
