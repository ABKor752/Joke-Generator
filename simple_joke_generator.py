from transformers import AdamW, AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, GPT2Tokenizer, GPT2LMHeadModel, get_scheduler, BartTokenizer, BartForConditionalGeneration, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import pprint
import evaluate
import random
import argparse

SEED=595

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_NAME="facebook/bart-base"

# tokenizer = AutoTokenizer.from_pretrained('t5-base')
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_data(train_file, test_file):
    def tokenize_function(examples):
        tokenized_examples = tokenizer(examples['body'], text_target=examples['punchline'], padding='max_length', truncation=True, return_tensors='pt')
        return tokenized_examples
    
    funny_train = load_dataset("csv", data_files=train_file, delimiter='\t', split='train[:92%]')
    funny_val = load_dataset("csv", data_files=train_file, delimiter='\t', split='train[92%:]')
    funny_test = load_dataset("csv", data_files=test_file, delimiter='\t', split='train')

    tokenized_train = funny_train.map(tokenize_function, batched=True)
    tokenized_train = tokenized_train.remove_columns(["body", "punchline"])
    tokenized_train.set_format("torch")

    tokenized_val = funny_val.map(tokenize_function, batched=True)
    tokenized_val = tokenized_val.remove_columns(["body", "punchline"])
    tokenized_val.set_format("torch")

    tokenized_test = funny_test.map(tokenize_function, batched=True)
    tokenized_test.set_format("torch")

    return tokenized_train, tokenized_val, tokenized_test

def main(params):
    train_file = params.train_file
    test_file = params.test_file
    tokenized_train, tokenized_val, tokenized_test = load_data(train_file, test_file)
#     model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    num_train_epochs = 15
    num_training_steps = num_train_epochs * len(tokenized_train)
    optimizer = AdamW(model.parameters())
    # TODO: fiddle with the type of scheduler and see if results improve (try linear)
    #lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results1",
        evaluation_strategy="epoch",
        learning_rate=6e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        weight_decay=0.02,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
#        optimizers=(optimizer,lr_scheduler),
    )
    
    trainer.train()

    metric = evaluate.load("bleu")
    model.eval()
    all_predictions = []

    torch.save(model.state_dict(), params.model_file)

    # TODO: unsure if we need collate_fn argument here
    test_dataloader = DataLoader(tokenized_test, batch_size=1)
    for batch in test_dataloader:
        batch_input = {k: v.to(device) for k, v in batch.items() if not isinstance(v, list)}
        with torch.no_grad():
            outputs = model(**batch_input)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        predictions = [tokenizer.decode(prediction, skip_special_tokens=True).replace('\n', '').replace('.', '') for prediction in predictions]
        all_predictions.extend(list(predictions))
        references = [[actual.replace('\n', '').replace('.', '')] for actual in batch["punchline"]]
        print('Body: ', batch["body"])
        print('Prediction: ', predictions)
        print('Reference: ', references)
        print()
        metric.add_batch(predictions=predictions, references=references)

    score = metric.compute()
    print('Bleu score: ', score['bleu'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate the joke generator model')
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--model_file', type=str, default="")
    main(parser.parse_args())

