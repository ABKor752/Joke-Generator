from transformers import AdamW, AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, GPT2Tokenizer, GPT2LMHeadModel, get_scheduler, BartTokenizer, BartForConditionalGeneration, get_cosine_schedule_with_warmup
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import pprint
import evaluate
import argparse

SEED=595

MODEL_NAME="facebook/bart-base"

# tokenizer = AutoTokenizer.from_pretrained('t5-base')
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_data(train_file, test_file):
    def tokenize_function(examples):
        tokenized_examples = tokenizer(examples['body'], text_target=examples['punchline'], padding='max_length', truncation=True, return_tensors='pt')
        return tokenized_examples
    
    funny_train = load_dataset("csv", data_files=train_file, delimiter='\t', split='train')
    funny_test = load_dataset("csv", data_files=test_file, delimiter='\t', split='train')

    tokenized_train = funny_train.map(tokenize_function, batched=True)
    tokenized_train = tokenized_train.remove_columns(["body", "punchline"])
    tokenized_train.set_format("torch")

    tokenized_test = funny_test.map(tokenize_function, batched=True)
    tokenized_test.set_format("torch")

    return tokenized_train, tokenized_test
