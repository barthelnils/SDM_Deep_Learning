import numpy as np
from sklearn.metrics import roc_auc_score
from modules.data_prep import load_data 

def calculate_permutation_importance(model, mean, std, base_dir, month, year):
    feature_names = ['Bedrock', 'Distance', 'Slope', 'Meanchla', 'Meansst', 'Salinity']

    num_iterations = 50
    env_data, labels = load_data(base_dir, year, month)
    X_test_norm = (env_data - mean) / std
    y_test = labels

    importance_scores = np.zeros(X_test_norm.shape[1])
    original_auc = roc_auc_score(y_test, model.predict(X_test_norm))
    print(f"Calculating Permutation Importance for {month} {year}")
    for _ in range(num_iterations):
        iteration_scores = np.zeros(X_test_norm.shape[1])
        for feature_index in range(X_test_norm.shape[1]):
            X_test_permuted = X_test_norm.copy()
            permuted_values = np.random.permutation(X_test_permuted[:, feature_index])
            X_test_permuted[:, feature_index] = permuted_values
            y_pred_permuted = model.predict(X_test_permuted, verbose=0)
            permuted_auc = roc_auc_score(y_test, y_pred_permuted)
            iteration_scores[feature_index] = original_auc - permuted_auc
        importance_scores += iteration_scores

    importance_scores /= num_iterations

    # Ensure importance scores are non-negative
    importance_scores = np.maximum(importance_scores, 0)

    # Normalize to range between 0 and 100
    max_importance = np.sum(importance_scores)
    if max_importance > 0:
        percent_contributions = (importance_scores / max_importance) * 100
    else:
        percent_contributions = np.zeros_like(importance_scores)

    for i, score in enumerate(percent_contributions):
        print(f"{feature_names[i]} percent contribution: {score:.2f}%")

    return percent_contributions
