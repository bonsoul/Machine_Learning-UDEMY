from transformers import BertTokenizer, BertModel

# Load pretrained tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Tokenize a sample input
text = "Transformers are powerful models for NLP tasks"
inputs = tokenizer(text, return_tensors="pt")  # note: return_tensors

# Pass the input through the model
outputs = model(**inputs)

print("Hidden States Shape:", outputs.last_hidden_state.shape)
