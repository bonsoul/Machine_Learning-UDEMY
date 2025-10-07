from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


#load the dataset
dataset = load_dataset("imdb")

tokenizer = AutoTokenizer.from_pretrained()