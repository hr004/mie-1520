from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModel
import re
from sklearn.svm import SVC
from sklearn.metrics import classification_report

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
feature_extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer)


def extract_features(batch):
    features = feature_extractor(batch["text"], return_tensors="pt")
    mean_pooling = list(
        map(lambda x: list(x.squeeze(0).numpy().mean(axis=0)), features)
    )
    return {"cls": mean_pooling}


def get_dataset():

    ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
    ds = ds.map(lambda x: {"text": re.sub(r"(\s)@\w+", "", x["text"])})

    split = ds.train_test_split(test_size=0.1)
    training_set = split["train"]
    eval_set = split["test"]

    training_set = training_set.map(extract_features, batched=True, batch_size=16)
    eval_set = eval_set.map(extract_features, batched=True, batch_size=16)
    return training_set, eval_set


def main():
    training_set, eval_set = get_dataset()
    svm = SVC()
    svm.fit(training_set["cls"], training_set["label"])
    preds = svm.predict(eval_set["cls"])
    print(classification_report(eval_set["label"], preds))


if __name__ == "__main__":
    main()
