import os
import yaml
import pandas as pd
import numpy as np
from modules.data_prep import load_data
from modules.model_training import train_model, save_model
from modules.evaluation import save_auc_values
from modules.sdm_creation import create_sdm
from modules.permutation_importance import calculate_permutation_importance
from modules.projection import project_model



def main():
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    base_dir = config['base_dir']
    model_dir = config['model_dir']
    output_dir = config['output_dir']
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    training_months = config['training_months']
    years = config['years']
    projection_year = config['projection_year']
    monthly_name_pattern = config['monthly_name_pattern']
    csv_file_name = config['csv_file_name']
    final_model_name = config['final_model_name']
    perm_csv_file_name = config['perm_csv_file_name']
    
    model = None
    prev_month_model_path = None
    mean_auc_values = []

    all_importance_scores = []
    
    
    for year in years:
        for month in training_months:
            env_data, labels = load_data(base_dir, year, month)
            model, auc_score, mean, std = train_model(env_data, labels, model, prev_month_model_path, config)
            global_mean, global_std = mean, std 
            model_path = os.path.join(model_dir, monthly_name_pattern.format(month=month, year=year))
            save_model(model, model_path)
            mean_auc_values.append((month, year, auc_score))
            prev_month_model_path = model_path

            # Create SDM for the current month
            create_sdm(base_dir, model_dir, output_dir, year, month, global_mean, global_std)

            # Calculate permutation importance for the current month
            importance_scores = calculate_permutation_importance(model, mean, std, base_dir, month, year)
            all_importance_scores.append(importance_scores)

    save_auc_values(mean_auc_values, output_dir, csv_file_name)
    final_model_path = os.path.join(model_dir, final_model_name)
    save_model(model, final_model_path)

    # Calculate mean permutation importance across all months
    mean_importance_scores = np.mean(all_importance_scores, axis=0)

    # Save mean permutation importance
    feature_names = ['Bedrock', 'Distance', 'Slope', 'Meanchla', 'Meansst', 'Salinity']
    permuted_csv_file = os.path.join(config['output_dir'], perm_csv_file_name)
    df = pd.DataFrame([mean_importance_scores], columns=feature_names)
    df.to_csv(permuted_csv_file, index=False)
    print("Mean permutation scores saved!")

    # Project model onto specified projection year data
    project_model(config, projection_year)

if __name__ == '__main__':
    main()
