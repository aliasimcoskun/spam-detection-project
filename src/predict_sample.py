import os
import pickle
from preprocessing import preprocess_text

# Paths for test directory, feature file, and model file
test_dir = './test_emails'
features_path = './data/features.pkl'
model_path = './models/optimized_svm_model.pkl'  # Assume the best optimized model is saved here

# Load the vectorizer
with open(features_path, 'rb') as f:
    data = pickle.load(f)
vectorizer = data['vectorizer']

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Read, preprocess, and predict on test emails
for filename in os.listdir(test_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(test_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as email_file:
            content = email_file.read()

        # Preprocess the email content
        clean_content = preprocess_text(content)
        
        # Convert text to feature vector
        X_test = vectorizer.transform([clean_content])
        
        # Convert sparse matrix to dense array
        X_test_dense = X_test.toarray()
        
        # Make prediction
        prediction = model.predict(X_test_dense)[0]
        label = 'SPAM' if prediction == 1 else 'HAM'

        print(f"{filename} → Prediction: {label}")