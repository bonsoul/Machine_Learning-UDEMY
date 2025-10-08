from transformers import BertTokenizer, TFBertModel

# Load pretrained tokenizer and model (TensorFlow)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertModel.from_pretrained("bert-base-uncased", from_pt=True)  # <-- important

# Tokenize input
text = "Transformers are powerful models for NLP tasks"
inputs = tokenizer(text, return_tensors="tf")

# Forward pass
outputs = model(**inputs)

print("Hidden States Shape:", outputs.last_hidden_state.shape)
