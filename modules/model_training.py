import numpy as np
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

def train_model(env_data, labels, model, prev_month_model_path, config):
    model_config = config['model_config']
    auc_scores = []
    mean = None
    std = None

    positive_samples = env_data[labels == 1]
    n_folds = min(10, len(positive_samples))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_index, test_index) in enumerate(skf.split(env_data, labels)):
        print(f"Processing fold {fold + 1}/{n_folds}")

        X_train, X_test = env_data[train_index], env_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Handle NaN values
        nan_rows_train = np.argwhere(np.isnan(X_train))
        nan_rows_test = np.argwhere(np.isnan(X_test))
        X_train = np.delete(X_train, nan_rows_train[:, 0], axis=0)
        y_train = np.delete(y_train, nan_rows_train[:, 0], axis=0)
        X_test = np.delete(X_test, nan_rows_test[:, 0], axis=0)
        y_test = np.delete(y_test, nan_rows_test[:, 0], axis=0)

        # Normalize data
        if mean is None or std is None:
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        # Build or load model
        if model is None:
            model = keras.Sequential()
            model.add(keras.layers.Input(shape=(6,)))
            for layer in model_config['dense_layers']:
                units = layer['units']
                activation = layer['activation']
                kernel_regularizer = keras.regularizers.l1(layer['regularization_strength']) if layer['kernel_regularizer'] == 'l1' else None
                model.add(keras.layers.Dense(units, activation=activation, kernel_regularizer=kernel_regularizer))
            model.add(keras.layers.Dense(1, activation='sigmoid'))
        
        # Compile the model
        optimizer = keras.optimizers.get(model_config['optimizer'])
        model.compile(optimizer=optimizer, loss=model_config['loss'], metrics=model_config['metrics'])
        
        # Load previous month's model if specified
        if prev_month_model_path:
            model = keras.models.load_model(prev_month_model_path)
            model.compile(optimizer=optimizer, loss=model_config['loss'], metrics=model_config['metrics'])  # Re-compile after loading
        
        # Train model
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=model_config['early_stopping_patience'], restore_best_weights=True)
        history = model.fit(X_train_norm, y_train, epochs=model_config['epochs'], validation_data=(X_test_norm, y_test),
                            batch_size=model_config['batch_size'], callbacks=[early_stopping], verbose=0)

        # Evaluate model
        y_pred = model.predict(X_test_norm, verbose=0)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)

    # Calculate mean AUC across all folds
    mean_auc = np.mean([score for score in auc_scores if not np.isnan(score)])
    print(f"Mean AUC for current month and year: {mean_auc}")

    return model, mean_auc, mean, std

def save_model(model, path):
    model.save(path)
