# 📧 Spam Detection Project

## 📝 Project Description
This project implements a machine learning-based email spam detection system that can automatically classify incoming emails as either spam or legitimate (ham). The system uses Natural Language Processing (NLP) techniques and multiple machine learning algorithms to achieve high accuracy in spam detection.

## 🎯 Key Features
- Text preprocessing and cleaning
- Feature extraction using TF-IDF vectorization
- Multiple machine learning models (Naive Bayes and SVM)
- Model optimization with GridSearchCV
- Performance evaluation with confusion matrices and ROC curves
- Easy-to-use prediction interface for new emails

## 📑 Contents
- [Project Description](#-project-description)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Technical Requirements](#️-technical-requirements)
- [Installation Guide](#️-installation-guide)
  - [Clone the Repository](#1️⃣-clone-the-repository)
  - [Set Up Virtual Environment](#2️⃣-set-up-virtual-environment)
  - [Install Dependencies](#3️⃣-install-dependencies)
  - [Download and Prepare Dataset](#4️⃣-download-and-prepare-dataset)
- [Usage Guide](#-usage-guide)
  - [Data Processing Pipeline](#1️⃣-data-processing-pipeline)
  - [Model Training and Optimization](#2️⃣-model-training-and-optimization)
  - [Making Predictions](#3️⃣-making-predictions)
- [Performance Metrics](#-performance-metrics)
- [Dependencies](#-dependencies)
- [Contributing](#-contributing)
- [License](#-license)

## Project Structure
```
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
```

## 🛠️ Technical Requirements
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository
    ```
    git clone https://github.com/aliasimcoskun/spam_detection_project.git
    cd spam_detection_project
    ```

### 2️⃣ Set Up Virtual Environment
    ```
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3️⃣ Install Dependencies
    ```
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

### 4️⃣ Download and Prepare Dataset
    ```
    mkdir -p data/spam data/ham
    wget https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2
    wget https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2
    tar -xjf 20021010_spam.tar.bz2 -C data/spam/
    tar -xjf 20021010_easy_ham.tar.bz2 -C data/ham/
    ```

## 🚀 Usage Guide

### 1️⃣ Data Processing Pipeline
    ```
    # Step 1: Load and organize email data
    python src/data_loader.py
    ```
    Creates `data/processed_data.csv`.
    ```
    # Step 2: Clean and preprocess text data
    python src/preprocessing.py
    ```
    Creates `data/processed_data_clean.csv`.
    ```
    # Step 3: Extract features using TF-IDF
    python src/feature_extraction.py
    ```
    Creates `data/features.pkl`.

### 2️⃣ Model Training and Optimization
    ```
    # Train basic models (Naive Bayes and SVM)
    python src/model_training.py
    ```
    Trains Naive Bayes and SVM models and saves them in `models/`.
    ```
    # Optimize SVM model using GridSearchCV
    python src/model_optimization.py
    ```
    Optimizes the SVM model and saves the best model.
    ```
    # Evaluate model performance
    python src/evaluation.py
    ```
    Visualizes confusion matrices and ROC curves.

### 3️⃣ Making Predictions
    ```
    # Predict on sample emails
    python src/predict_sample.py
    ```
    Predicts whether sample emails in the `test_emails/` directory are spam or ham.

## 📊 Performance Metrics

The project generates several visualization files:

- Confusion matrices for each model
- ROC curves showing model performance
- Detailed evaluation metrics including:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

## 📦 Dependencies
Core libraries used in this project:

- pandas (≥ 1.0.0) - Data manipulation
- scikit-learn (≥ 0.24.0) - Machine learning algorithms
- nltk (≥ 3.5) - Natural language processing
- spacy (≥ 3.0.0) - Advanced text processing
- matplotlib (≥ 3.3.0) - Visualization
- seaborn (≥ 0.11.0) - Statistical visualization

## 🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.