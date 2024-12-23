import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

# Get the current working directory
current_directory = os.getcwd()

# Step 1: Load data (assuming the data is a CSV)
df = pd.read_csv('data.csv')  
df = df[['input', 'label']]  

# Step 2: Preprocess data 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['input'], padding="max_length", truncation=True, max_length=128)

# Step 3: Prepare dataset for fine-tuning
# Split the data into training and test sets (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Step 4: Check if a pre-trained model exists, if not, train a new one
model_dir = current_directory + '/xss_model'

if os.path.exists(model_dir):
    # Load the pre-trained model and tokenizer if available
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    print("Loaded pre-trained model.")
else:
    # Train the model if it's not available
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=current_directory + '/results',          
        num_train_epochs=3,             
        per_device_train_batch_size=8,   
        per_device_eval_batch_size=16,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir=current_directory + '/logs',            # directory for storing logs in the current directory
        logging_steps=10,
        evaluation_strategy="epoch",     
    )

    # Create Trainer
    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=test_dataset,           
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer in the current directory
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("Model trained and saved.")

# Step 9: Evaluate the model
trainer = Trainer(model=model)  # Create a new trainer for evaluation
results = trainer.evaluate(test_dataset)
print("Evaluation Results:", results)

# Step 10: Use the model for prediction
def predict(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

# Example usage of prediction
input_text = "<script>alert('XSS')</script>"
input_text2 = "Mahmoud+Amine"
prediction = predict(input_text)
prediction2 = predict(input_text2)
print("Prediction :", prediction)
print("Prediction :", prediction2)