from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch.nn.functional as F
import evaluate
import numpy as np
import torch
from torch.utils.data import Dataset

import csv
import random
from sklearn.model_selection import train_test_split


MODEL="bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
metric = evaluate.load("accuracy")
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# TODO: aggregate and dataset-ify the reddit training data
def get_datasets():
    def add_data(path, lst, label):
        with open(path, "r") as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            next(tsv_reader)
            for row in tsv_reader:
                lst.append((row[0] + ' ' + row[1], label))
    train, test = [], []
    PATH_TO_FUNNY_TRAIN = "../datasets/data/reddit_preprocessed/funny.tsv"
    PATH_TO_FUNNY_TEST = "../datasets/data/reddit_preprocessed/test_funny.tsv"
    PATH_TO_UNFUNNY_TRAIN = "../datasets/data/reddit_preprocessed/unfunny.tsv"
    PATH_TO_UNFUNNY_TEST = "../datasets/data/reddit_preprocessed/test_unfunny.tsv"

    add_data(PATH_TO_FUNNY_TRAIN, train, 1)
    add_data(PATH_TO_FUNNY_TEST, test, 1)
    add_data(PATH_TO_UNFUNNY_TRAIN, train, 0)
    add_data(PATH_TO_UNFUNNY_TEST, test, 0)

    random.shuffle(train)
    random.shuffle(test)

    train_jokes, train_labels = [joke for joke, label in train], [label for joke, label in train] 
    test_jokes, test_labels = [joke for joke, label in test], [label for joke, label in test] 
    train_jokes, val_jokes, train_labels, val_labels = train_test_split(train_jokes, train_labels, test_size=.2)

    train_encodings = tokenizer(train_jokes, truncation=True, padding=True)
    val_encodings = tokenizer(val_jokes, truncation=True, padding=True)
    test_encodings = tokenizer(test_jokes, truncation=True, padding=True)
    return train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels

# may need some printing to check hwat's going on here but hopefully it just works lol
class JokeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocess_function(examples):
    return tokenizer(examples["joke"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # check that this is the way to do it...
    # print(predictions)
    # print(F.softmax(predictions, dim=-1))

    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels = get_datasets()
    train = JokeDataset(train_encodings, train_labels)
    val = JokeDataset(val_encodings, val_labels)
    test = JokeDataset(test_encodings, test_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="results",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=7,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("model.pt")

if __name__ == '__main__':
    main()

"""
inference should look like this

classifier = pipeline("sentiment-analysis", model="model.pt")
classifier(JokeDataset)

can alternatively copy the format from hw3 if huggingface doesn't let us classify on a whole dataset object
"""