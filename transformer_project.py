from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer


#load dataset fro summarization
dataset = load_dataset("cnn_dailymail", "3.0.0")
print(dataset["train"][0])

#for translation
#dataset1 = load_dataset("wmt14", "en-fr")

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")


#tokenize for summarization

def tokenize_function(examples):
    inputs = ["summarize: " + doc for doc in examples["articles"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    
    # tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=150, truncation=True)
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datsets = dataset.map(tokenize_function,batched=True)


model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")


training_args = TrainingArguments(
    output_dir = "./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True
)


trainer = Trainer(
    model= model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer
)

trainer.train()


sample_text = "The Transformer model has revolutionized NLP by enabling paraller processing"
inputs = tokenizer("summarizer: "+ sample_text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)


print("generated Summary: ", tokenizer.decode(outputs[0], skip_special_token=true))

#metric = load_metric("rouge") #sacrenleu
#predictions = outputs["generated_text"]
#references = dataset["validation"]["highlights"]


#results = metric.compute(predictions=predictions, references=refences)
#print(results)