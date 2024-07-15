import csv
import os

def save_auc_values(mean_auc_values, model_dir, csv_file_name):
    csv_file = os.path.join(model_dir, csv_file_name)
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Month', 'Year', 'Mean AUC'])
        for row in mean_auc_values:
            writer.writerow(row)
