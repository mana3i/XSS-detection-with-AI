from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from flask_cors import CORS

# Load the model and tokenizer
model_dir = './xss_model'
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)


app = Flask(__name__)
CORS(app)

def predict(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()  
    if 'text' not in data:
        return jsonify({"error": "No text field provided"}), 400

    text = data['text']
    prediction = predict(text)
    print(f"Input: {text}, Prediction: {prediction}")
    return jsonify({"text": text, "prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
