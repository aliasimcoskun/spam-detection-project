import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{model_name}_roc_curve.png")
    plt.close()

def evaluate_model(model_path, features_pkl, model_name):
    # Load the model and data
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(features_pkl, 'rb') as file:
        data = pickle.load(file)
    X = data['X']
    y = data['y']
    
    # Split the data into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Generate and save the confusion matrix plot
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    # Calculate scores and plot the ROC curve
    if hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_scores, model_name)

if __name__ == "__main__":
    features_pkl = './data/features.pkl'
    models_dir = './models'
    
    # Evaluate Naive Bayes Model
    nb_model_path = f"{models_dir}/naive_bayes_model.pkl"
    evaluate_model(nb_model_path, features_pkl, "Naive Bayes")
    
    # Evaluate SVM Model
    svm_model_path = f"{models_dir}/svm_model.pkl"
    evaluate_model(svm_model_path, features_pkl, "SVM")
    
    # Evaluate Optimized SVM Model
    optimized_svm_model_path = f"{models_dir}/optimized_svm_model.pkl"
    evaluate_model(optimized_svm_model_path, features_pkl, "Optimized SVM")