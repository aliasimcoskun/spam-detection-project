import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_models(features_pkl, models_dir):
    # Load features and labels from the pickle file
    with open(features_pkl, 'rb') as file:
        data = pickle.load(file)
    X = data['X']
    y = data['y']
    
    # Check for NaN values in labels
    if np.isnan(y).any():
        raise ValueError("Input y contains NaN.")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    
    # Evaluate model performance
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    precision_nb = precision_score(y_test, y_pred_nb)
    recall_nb = recall_score(y_test, y_pred_nb)
    f1_nb = f1_score(y_test, y_pred_nb)
    
    print("Naive Bayes Model Performance:")
    print(f"Accuracy: {accuracy_nb:.4f}")
    print(f"Precision: {precision_nb:.4f}")
    print(f"Recall: {recall_nb:.4f}")
    print(f"F1 Score: {f1_nb:.4f}\n")
    
    # Save the Naive Bayes model to the specified directory
    nb_model_path = f"{models_dir}/naive_bayes_model.pkl"
    with open(nb_model_path, 'wb') as file:
        pickle.dump(nb_model, file)
    print(f"Naive Bayes model saved as {nb_model_path}.")
    
    # Train the SVM model with a linear kernel
    svm_model = svm.SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    
    # Evaluate model performance
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm)
    recall_svm = recall_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)
    
    print("SVM Model Performance:")
    print(f"Accuracy: {accuracy_svm:.4f}")
    print(f"Precision: {precision_svm:.4f}")
    print(f"Recall: {recall_svm:.4f}")
    print(f"F1 Score: {f1_svm:.4f}\n")
    
    # Save the SVM model to the specified directory
    svm_model_path = f"{models_dir}/svm_model.pkl"
    with open(svm_model_path, 'wb') as file:
        pickle.dump(svm_model, file)
    print(f"SVM model saved as {svm_model_path}.")

if __name__ == "__main__":
    features_pkl = './data/features.pkl'
    models_dir = './models'
    train_models(features_pkl, models_dir)