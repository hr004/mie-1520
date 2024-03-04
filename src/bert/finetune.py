from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np

from datasets import load_dataset
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=3
)
metric = evaluate.load("accuracy")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def get_dataset(split_size=0.1):
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
    ds = ds.map(lambda x: {"text": re.sub(r"(\s)@\w+", "", x["text"])})
    ds = ds.map(tokenize_function, batched=True, batch_size=16)
    dataset_split = ds.train_test_split(test_size=split_size)
    train_dataset = dataset_split["train"].shuffle(seed=42)
    eval_dataset = dataset_split["test"].shuffle(seed=42)
    return train_dataset, eval_dataset


def main(num_epochs=3):
    train_dataset, eval_dataset = get_dataset()
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        logging_dir="./logs",
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    print(eval_results)

if __name__ == "__main__":
    main(num_epochs=6)