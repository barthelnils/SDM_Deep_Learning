# Deep Learning Project for Species Distribution Modeling (SDM)

## Overview
This project trains deep learning models to predict species distribution based on environmental data. It includes scripts for training, evaluation, permutation importance analysis, and projecting models onto future data.

## Directory Structure
- `config.yaml`: Configuration file containing paths and parameters.
- `main.py`: Main script to run the entire process.
- `modules/`: Directory containing separate modules for different functionalities.
  - `data_preparation.py`: Functions to load data.
  - `model_training.py`: Functions to initialize, train, and save models.
  - `evaluation.py`: Functions to save AUC values.
  - `sdm_creation.py`: Functions to create SDM raster files.
  - `permutation_importance.py`: Functions to calculate permutation importance.
  - `projection.py`: Functions to project trained models onto future data.
- `requirements.txt`: List of required Python packages.
- `README.md`: Documentation of the project.

## Usage
1. **Setup**: Modify `config.yaml` with your paths and parameters.
2. **Install Dependencies**: Run `pip install -r requirements.txt`.
3. **Run Main Script**: Execute `python main.py` to train models, create SDMs, calculate permutation importance, and project models.

## Detailed Steps
1. **Data Preparation**: Loads environmental data and occurrence points.
2. **Model Training**: Trains models for each month and year combination.
3. **Evaluation**: Saves AUC values for each trained model.
4. **SDM Creation**: Generates SDM raster files for each month.
5. **Permutation Importance**: Calculates permutation importance for environmental variables.
6. **Projection**: Projects the trained model onto 2022 data.

## Author
Nils Barthel
