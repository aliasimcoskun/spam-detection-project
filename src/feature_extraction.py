import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def extract_features(input_csv, output_pkl, max_features=5000):
    df = pd.read_csv(input_csv)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df['label'].map({'spam':1, 'ham':0}).values
    # Save the feature matrix, labels, and vectorizer
    with open(output_pkl, 'wb') as file:
        pickle.dump({'X': X, 'y': y, 'vectorizer': vectorizer}, file)
    print(f"Feature extraction completed. Data saved as {output_pkl}.")

if __name__ == "__main__":
    input_csv = './data/processed_data_clean.csv'
    output_pkl = './data/features.pkl'
    extract_features(input_csv, output_pkl)