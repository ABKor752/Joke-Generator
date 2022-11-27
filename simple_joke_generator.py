from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, GPT2Tokenizer, GPT2LMHeadModel, get_scheduler
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import pprint
import evaluate

SEED=595

# tokenizer = AutoTokenizer.from_pretrained('t5-base')
tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_data():
    def tokenize_function(examples):
        tokenized_examples = tokenizer(examples['body'], text_target=examples['punchline'], padding='max_length', truncation=True, return_tensors='pt')
        return tokenized_examples
    
    funny_train = load_dataset("csv", data_files="datasets/data/reddit_preprocessed/funny.tsv", delimiter='\t', split='train[1:2]')
    funny_test = load_dataset("csv", data_files="datasets/data/reddit_preprocessed/test_funny.tsv", delimiter='\t', split='train[1:2]')

    tokenized_train = funny_train.map(tokenize_function, batched=True)
    tokenized_train = tokenized_train.remove_columns(["body", "punchline"])
    tokenized_train.set_format("torch")

    tokenized_test = funny_test.map(tokenize_function, batched=True)
    tokenized_test.set_format("torch")

    return tokenized_train, tokenized_test

def main():
    tokenized_train, tokenized_test = load_data()
#     model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
    model = AutoModelForSeq2SeqLM.from_pretrained("RUCAIBox/mvp")
    model.to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()

    metric = evaluate.load("bleu")
    model.eval()
    all_predictions = []

    # TODO: unsure if we need collate_fn argument here
    test_dataloader = DataLoader(tokenized_test, batch_size=1)
    for batch in test_dataloader:
        batch_input = {k: v.to(device) for k, v in batch.items() if not isinstance(v, list)}
        with torch.no_grad():
            outputs = model(**batch_input)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        predictions = [tokenizer.decode(prediction) for prediction in predictions]
        # TODO: truncate padding tokens
        all_predictions.extend(list(predictions))
        metric.add_batch(predictions=predictions, references=batch["punchline"])

    score = metric.compute()
    print('Bleu score: ', score['bleu'])

if __name__ == '__main__':
    main()

