from flask import Flask, render_template, request, jsonify
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# Load necessary models
label_encoder = joblib.load('label_encoder.joblib')
# Load tokenizer from local files
tokenizer = BertTokenizer.from_pretrained('./')
# Load related model from local files
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)
model.load_state_dict(torch.load('personality_model.pth', map_location=torch.device('cpu')))
model.eval()

# Create Flask App
app = Flask(__name__)

# Index page
@app.route('/')
def home():
    return render_template('index.html')

# Personality type to animal mapping
personality_to_animal = {
    'E': 'dolphin',
    'I': 'owl',
    'S': 'elephant',
    'N': 'eagle',
    'T': 'wolf',
    'F': 'dog',
    'J': 'bee',
    'P': 'fox'
}

# Receive data from form and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        # Convert data to tokens
        inputs = tokenizer(user_input, return_tensors='pt')
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            predicted_probability = probabilities[0][predicted_class].item()
        # Decode the predicted class
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        # Map personality type to animal
        predicted_animal = personality_to_animal.get(predicted_label, 'Unknown')
        return jsonify({
            'prediction': predicted_label,
            'probability': f"{predicted_probability * 100:.2f}%",
            'animal': predicted_animal
        })

if __name__ == '__main__':
    app.run(debug=True)