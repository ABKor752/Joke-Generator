from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, GPT2Tokenizer, GPT2LMHeadModel, get_scheduler
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import pprint
import evaluate

SEED=595

tokenizer = AutoTokenizer.from_pretrained("t5-base")
# TODO: switch to MVP for Great Lakes
# tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp")
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# print('\nvocab size: ' +  str(tokenizer.vocab_size) + '\n' )

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_data():
    def tokenize_function(examples):
        tokenized_examples = tokenizer(examples['body'], text_target=examples['punchline'], padding='max_length', truncation=True, return_tensors='pt')
        # print(tokenized_examples.keys())
        return tokenized_examples


    # funny_data_files = {"train": "funny_train.tsv", "test" : "funny_val.tsv"}
    # blergh = load_dataset("../datasets/data/reddit_full", data_files=funny_data_files)
    funny_train = load_dataset("csv", data_files="datasets/data/reddit_preprocessed/funny.tsv", delimiter='\t', split='train[5:10]')
    funny_test = load_dataset("csv", data_files="datasets/data/reddit_preprocessed/test_funny.tsv", delimiter='\t', split='train[5:10]')
    
    # funny_train = load_dataset("csv", data_files="datasets/data/reddit_preprocessed/funny.tsv", delimiter='\t', split='train')
    # funny_test = load_dataset("csv", data_files="datasets/data/reddit_preprocessed/test_funny.tsv", delimiter='\t', split='train')
    
    # print(funny_train, funny_test)


    tokenized_train = funny_train.map(tokenize_function, batched=True)
    # tokenized_train = tokenized_train.remove_columns(["body", "punchline"])
    tokenized_train.set_format("torch")

    tokenized_test = funny_test.map(tokenize_function, batched=True)
    # tokenized_test = tokenized_test.remove_columns(["body", "punchline"])
    tokenized_test.set_format("torch")

    # train_dataloader = DataLoader(tokenized_datasets["train"].shuffle(seed=SEED), shuffle=True, batch_size=500)
    # test_dataloader = DataLoader(tokenized_datasets["train"], batch_size=500) #TODO: combine tokenized datasets and dataloaders?
    # print(train_dataloader)
    # print(tokenized_test['body'])
    print(tokenized_test)
    return tokenized_train, tokenized_test

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
    tokenized_train, tokenized_test = load_data()
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    # TODO: switch to MVP for Great Lakes
    # model = AutoModelForSeq2SeqLM.from_pretrained("RUCAIBox/mvp")
    # print(model)
    model.to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        # TODO: increase for Great Lakes
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=3,
        # num_train_epochs=2,
        num_train_epochs=1,
        # removing the below because this crashes if CUDA is not set up in a specific manner
        # fp16=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        # TODO: put an actual eval dataset here that isn't same as test
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()

    metric = evaluate.load("bleu")

    # print(tokenized_test['body'])
    # bodies = tokenized_test['body']
    # references = tokenized_test['punchline']
    # predictions = [model(body) for body in bodies]
    # results = metric.compute(predictions=predictions, references=references)
    model.eval()
    all_predictions = []

    # copy-pasted from hw3 and probably incorrect because of that
    # test_dataloader = DataLoader(tokenized_test, batch_size=1, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model))
    test_dataloader = DataLoader(tokenized_test, batch_size=1)
    for batch in test_dataloader:
        # print(type(batch))
        print(batch)
        # print(batch.items())
        # for k, v in batch.items():
        #    print(type(k), k)
        #    print(type(v), v)
        # batch = {k: v.to(device) for k, v in batch.items()}
        batch_input = {k: v.to(device) for k, v in batch.items() if not isinstance(v, list)}
        print(batch_input)
        # batch_references = [batch["punchline"]]
        with torch.no_grad():
            outputs = model(**batch_input)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        #print(outputs)
        #print(type(outputs))

        # logits = outputs.logits
        # predictions = torch.argmax(logits, dim=-1)
        predictions = [tokenizer.decode(prediction) for prediction in predictions]
        # TODO: truncate padding tokens
        all_predictions.extend(list(predictions))
        # this may have to be label instead
        metric.add_batch(predictions=predictions, references=batch["punchline"])

    score = metric.compute()
    # print('Test Accuracy:', score['accuracy'])
    print('Bleu score: ', score['bleu'])




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
