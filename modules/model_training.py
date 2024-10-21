import numpy as np  
import csv  
import os  
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold  
from sklearn.metrics import roc_curve, auc  

def train_model(env_data, labels, model, prev_month_model_path, config):
    """
    Trains a model using stratified k-fold cross-validation.

    Parameters:
    env_data (numpy array): Environmental data features for training.
    labels (numpy array): Labels corresponding to the data samples.
    model (keras.Model): Predefined model or None to create a new one.
    prev_month_model_path (str): Path to a model file to load weights from, if available.
    config (dict): Configuration dictionary for model settings.

    Returns:
    model: The trained Keras model.
    mean_auc: Mean AUC score across all folds.
    mean, std: Normalization parameters (mean and standard deviation).
    """
    
    model_config = config['model_config']  # Extract model configuration from the config
    auc_scores = []  # List to store AUC scores for each fold
    mean = None  # Mean of training data, used for normalization
    std = None  # Standard deviation of training data, used for normalization

    # Get the number of positive samples to determine the number of folds for cross-validation
    positive_samples = env_data[labels == 1]
    n_folds = min(10, len(positive_samples))  # Use up to 10 folds, but not more than the number of positive samples
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)  # Initialize stratified k-fold cross-validation

    # Loop through each fold of the cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(env_data, labels)):
        print(f"Processing fold {fold + 1}/{n_folds}")

        # Split the data into training and testing sets for this fold
        X_train, X_test = env_data[train_index], env_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Handle NaN values by removing rows with missing data
        nan_rows_train = np.argwhere(np.isnan(X_train))
        nan_rows_test = np.argwhere(np.isnan(X_test))
        X_train = np.delete(X_train, nan_rows_train[:, 0], axis=0)
        y_train = np.delete(y_train, nan_rows_train[:, 0], axis=0)
        X_test = np.delete(X_test, nan_rows_test[:, 0], axis=0)
        y_test = np.delete(y_test, nan_rows_test[:, 0], axis=0)

        # Normalize data using mean and standard deviation from the training data
        if mean is None or std is None:  # Calculate these values only once
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        # Build a new model if one is not provided
        if model is None:
            model = keras.Sequential()
            model.add(keras.layers.Input(shape=(6,)))  # Input layer with 6 features
            for layer in model_config['dense_layers']:
                units = layer['units']
                activation = layer['activation']
                kernel_regularizer = keras.regularizers.l1(layer['regularization_strength']) if layer['kernel_regularizer'] == 'l1' else None
                model.add(keras.layers.Dense(units, activation=activation, kernel_regularizer=kernel_regularizer))
            model.add(keras.layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification
        
        # Compile the model with the specified optimizer, loss, and metrics
        optimizer = keras.optimizers.get(model_config['optimizer'])
        model.compile(optimizer=optimizer, loss=model_config['loss'], metrics=model_config['metrics'])
        
        # Optionally load weights from a pre-trained model if specified
        if prev_month_model_path:
            model = keras.models.load_model(prev_month_model_path)
            model.compile(optimizer=optimizer, loss=model_config['loss'], metrics=model_config['metrics'])  # Re-compile after loading
        
        # Train the model with early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=model_config['early_stopping_patience'], restore_best_weights=True)
        history = model.fit(X_train_norm, y_train, epochs=model_config['epochs'], validation_data=(X_test_norm, y_test),
                            batch_size=model_config['batch_size'], callbacks=[early_stopping], verbose=0)

        # Predict probabilities on the test set and calculate the AUC score
        y_pred = model.predict(X_test_norm, verbose=0)
        fpr, tpr, _ = roc_curve(y_test, y_pred)  # Compute false positive rates and true positive rates
        auc_score = auc(fpr, tpr)  # Calculate the AUC score for this fold
        auc_scores.append(auc_score)

    # Calculate the mean AUC across all folds
    mean_auc = np.mean([score for score in auc_scores if not np.isnan(score)])
    print(f"Mean AUC for current month and year: {mean_auc}")

    return model, mean_auc, mean, std  # Return the model, mean AUC score, and normalization parameters

def save_model(model, path):
    """
    Saves the trained Keras model to the specified file path.
    
    Parameters:
    model (keras.Model): The Keras model to save.
    path (str): The file path where the model should be saved.
    """
    model.save(path)

def save_auc_values(mean_auc_values, model_dir, csv_file_name):
    """
    Saves the mean AUC values to a CSV file.

    Parameters:
    mean_auc_values (list): List of mean AUC values for different models.
    model_dir (str): Directory where the CSV file will be saved.
    csv_file_name (str): Name of the CSV file.
    """
    csv_file = os.path.join(model_dir, csv_file_name)  # Construct the full file path
    with open(csv_file, 'w', newline='') as file:  # Open the file in write mode
        writer = csv.writer(file)
        writer.writerow(['Month', 'Year', 'Mean AUC'])  # Write the header
        for row in mean_auc_values:  # Write each row of mean AUC values
            writer.writerow(row)
