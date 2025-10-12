from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 1️⃣ Load the dataset
dataset = load_dataset("imdb")

# 2️⃣ Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 3️⃣ Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4️⃣ Prepare the data
# Remove only the "text" column; keep "label" because the model needs it
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# Rename "label" to "labels" (Trainer expects this column name)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Set PyTorch tensor format
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# 5️⃣ Load the pretrained model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 6️⃣ Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",      # ✅ correct arg name (was eval_strategy)
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500
)

# 7️⃣ Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# 8️⃣ Train and evaluate
trainer.train()

results = trainer.evaluate()
print("Evaluation Results:", results)
