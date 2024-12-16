# Spam Detection Project

## Project Description
This project aims to perform spam detection in networked email systems using machine learning techniques.

## Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Project Structure

spam_detection_project/
│
├── data/
│   ├── spam/
│   ├── ham/
│   ├── features.pkl
│   ├── processed_data.csv
│   └── processed_data_clean.csv
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── model_optimization.py
│   ├── evaluation.py
│   └── predict_sample.py
│
├── models/
│   ├── naive_bayes_model.pkl
│   ├── svm_model.pkl
│   └── optimized_svm_model.pkl
│
├── reports/
│   └── project_report.pdf
│
├── test_emails/
│   ├── ham_test1.txt
│   └── spam_test1.txt
│
├── requirements.txt
└── README.md

## Setup

### Requirements
- Python 3.8 or above
- pip
- Virtualenv (optional)

### Steps
1. **Clone the Repository**
    ```
    git clone https://github.com/aliasimcoskun/spam_detection_project.git
    cd spam_detection_project
    ```

2. **Create and Activate a Virtual Environment**
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies**
    ```
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Download and Place the Dataset**
    ```
    mkdir -p data/spam data/ham
    wget https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2
    wget https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2
    tar -xjf 20021010_spam.tar.bz2 -C data/spam/
    tar -xjf 20021010_easy_ham.tar.bz2 -C data/ham/
    ```

## Usage

1. **Load Data:**
    ```
    python src/data_loader.py
    ```

    Creates `data/processed_data.csv`.

2. **Preprocess Data:**
    ```
    python src/preprocessing.py
    ```

    Creates `data/processed_data_clean.csv`.

3. **Feature Extraction:**
    ```
    python src/feature_extraction.py
    ```

    Creates `data/features.pkl`.

4. **Model Training:**
    ```
    python src/model_training.py
    ```

    Trains Naive Bayes and SVM models and saves them in `models/`.

5. **Model Optimization:**
    ```
    python src/model_optimization.py
    ```

    Optimizes the SVM model and saves the best model.

6. **Model Evaluation:**
    ```
    python src/evaluation.py
    ```

    Visualizes confusion matrices and ROC curves.

7. **Forecasting on Sample Emails:**
    ```
    python src/predict_sample.py
    ```

    Predicts whether sample emails in the `test_emails/` directory are spam or ham.

## Dependencies

- pandas
- numpy
- scikit-learn
- nltk
- spacy
- matplotlib
- seaborn
