from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import pprint

def load_data():
    def preprocess_function(examples):
        # body = [[body]]
        #tokenized_examples = tokenizer(body, punchline)
        pass
        # tokenize both and then 


    # funny_data_files = {"train": "funny_train.tsv", "test" : "funny_val.tsv"}
    # blergh = load_dataset("../datasets/data/reddit_full", data_files=funny_data_files)
    funny_train = load_dataset("csv", data_files="datasets/data/reddit_preprocessed/funny.tsv", delimiter='\t')
    print(funny_train)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(funny_train["train"][:2])

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