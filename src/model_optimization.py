import pickle
import time
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def optimize_svm(features_pkl, models_dir):
    start_time = time.time()
    print("Loading data...")
    # Load the data from the pickle file
    with open(features_pkl, 'rb') as file:
        data = pickle.load(file)
    X = data['X']
    y = data['y']
    print(f"Data loading completed. Time elapsed: {time.time() - start_time:.2f} seconds")

    # Split data into training and testing sets
    print("Splitting data into training and test sets...")
    start_split = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Splitting completed. Time elapsed: {time.time() - start_split:.2f} seconds")

    # Define the hyperparameter space for optimization
    parameters = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear'],
        'gamma': ['scale', 'auto']
    }
    print("Hyperparameter space defined:")
    print(parameters)

    # Perform Grid Search for hyperparameter tuning
    print("Starting Grid Search...")
    start_grid = time.time()
    svm_model = svm.SVC(probability=True)
    grid_search = GridSearchCV(
        svm_model,
        parameters,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=3
    )
    grid_search.fit(X_train, y_train)
    print(f"Grid Search completed. Time elapsed: {time.time() - start_grid:.2f} seconds")

    # Display the best parameters and score found by Grid Search
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")

    # Make predictions using the best model obtained
    print("Making predictions with the best model...")
    start_predict = time.time()
    best_svm = grid_search.best_estimator_
    y_pred_best_svm = best_svm.predict(X_test)
    print(f"Prediction completed. Time elapsed: {time.time() - start_predict:.2f} seconds")

    # Evaluate the performance of the optimized model
    print("Evaluating model performance...")
    accuracy_best_svm = accuracy_score(y_test, y_pred_best_svm)
    precision_best_svm = precision_score(y_test, y_pred_best_svm)
    recall_best_svm = recall_score(y_test, y_pred_best_svm)
    f1_best_svm = f1_score(y_test, y_pred_best_svm)

    print("Optimized SVM Model Performance:")
    print(f"Accuracy: {accuracy_best_svm:.4f}")
    print(f"Precision: {precision_best_svm:.4f}")
    print(f"Recall: {recall_best_svm:.4f}")
    print(f"F1 Score: {f1_best_svm:.4f}\n")

    # Save the optimized model to the specified directory
    print("Saving the model...")
    model_path = os.path.join(models_dir, 'optimized_svm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_svm, f)
    print(f"Model saved to: {model_path}")

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    features_pkl = './data/features.pkl'
    models_dir = './models'
    optimize_svm(features_pkl, models_dir)