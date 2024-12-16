import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove HTML tags from the text
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove special characters and numbers from the text
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Tokenize the text into individual words
    tokens = text.split()
    
    # Remove stop words and apply stemming to each word
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def preprocess_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['clean_text'] = df['text'].apply(preprocess_text)
    df.to_csv(output_csv, index=False)
    print(f"Preprocessing completed. Processed data saved as {output_csv}.")

if __name__ == "__main__":
    input_csv = './data/processed_data.csv'
    output_csv = './data/processed_data_clean.csv'
    preprocess_data(input_csv, output_csv)