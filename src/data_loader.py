import os
import pandas as pd

def load_emails_from_folder(folder, label):
    emails = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            # Process all files without checking the file extension
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                    if content.strip():  # Skip empty files
                        emails.append((label, content))
                    else:
                        print(f"Skipped empty file: {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return emails

def load_data(spam_dir, ham_dir):
    spam_emails = load_emails_from_folder(spam_dir, 'spam')
    ham_emails = load_emails_from_folder(ham_dir, 'ham')
    data = spam_emails + ham_emails
    df = pd.DataFrame(data, columns=['label', 'text'])
    return df

if __name__ == "__main__":
    # Specify dataset directories using full paths
    spam_dir = './data/spam/'
    ham_dir = './data/ham/'

    # Load data
    df = load_data(spam_dir, ham_dir)

    # Display the first few rows
    print(df.head())

    # Check label distribution
    print(df['label'].value_counts())

    # Save the dataset as CSV
    df.to_csv('./data/processed_data.csv', index=False)