from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, GPT2Tokenizer, GPT2LMHeadModel, get_scheduler
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import pprint
import evaluate

SEED=595

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_data():
    def tokenize_function(examples):
        tokenized_examples = tokenizer(examples['body'], examples['punchline'], padding='max_length', truncation=True, return_tensors='pt')
        return tokenized_examples


    # funny_data_files = {"train": "funny_train.tsv", "test" : "funny_val.tsv"}
    # blergh = load_dataset("../datasets/data/reddit_full", data_files=funny_data_files)
    funny_train = load_dataset("csv", data_files="datasets/data/reddit_preprocessed/funny.tsv", delimiter='\t')
    funny_test = load_dataset("csv", data_files="datasets/data/reddit_preprocessed/test_funny.tsv", delimiter='\t', split='test')
    # print(funny_train, funny_test)
    tokenized_datasets = funny_train.map(tokenize_function, batched=True)
    tokenized_datasets.set_format("torch")

    test_dataset = funny_test.map(tokenize_function, batched=True)
    test_dataset.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets["train"].shuffle(seed=SEED), shuffle=True, batch_size=500)
    test_dataloader = DataLoader(tokenized_datasets["train"], batch_size=500) #TODO: combine tokenized datasets and dataloaders?
    print(train_dataloader)
    return train_dataloader, test_dataloader

# def train(model, train_dataloader):
#     num_training_steps = 2 * len(train_dataloader)
#     optimizer = AdamW(model.parameters(), lr=1e-5)
    
#     lr_scheduler = get_scheduler(
#         name="linear", 
#         optimizer=optimizer, 
#         num_warmup_steps=0, 
#         num_training_steps=num_training_steps
#     )

#     progress_bar = tqdm(range(num_training_steps))

#     for epoch in range(2):
#         print('--- ' + str(epoch) + ' ---')
#         model.train()
#         for batch in train_dataloader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(**batch)
#             loss = outputs.loss
#             loss.backward()

#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
#             progress_bar.update(1)
#     return model

def main():
    train_dataloader, test_dataloader = load_data()
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model.to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        #per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        fp16=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader["train"],
        #eval_dataset=tokenized_books["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()

    metric = evaluate.load("bleu")
    bodies = test_dataloader['train']['body']
    references = test_dataloader['train']['punchline']
    predictions = [model(body) for body in bodies]
    results = metric.compute(predictions=predictions, references=references)
    print('Bleu score: ' + str(results.bleu))




if __name__ == '__main__':
    main()




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