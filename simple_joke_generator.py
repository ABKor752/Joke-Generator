from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import pprint

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def load_data():
    def tokenize_function(examples):
        tokenized_examples = tokenizer(examples['body'], examples['punchline'], padding='max_length', truncation=True)
        return tokenized_examples


    # funny_data_files = {"train": "funny_train.tsv", "test" : "funny_val.tsv"}
    # blergh = load_dataset("../datasets/data/reddit_full", data_files=funny_data_files)
    funny_train = load_dataset("csv", data_files="datasets/data/reddit_preprocessed/funny.tsv", delimiter='\t')
    tokenized_datasets = funny_train.map(tokenize_function, batched=True)
    print(tokenized_datasets)

if __name__ == '__main__':
    load_data()




"""
seq2seq example from https://huggingface.co/docs/transformers/v4.24.0/en/tasks/translation
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
"""