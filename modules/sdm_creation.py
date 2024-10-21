import os
import rasterio
import numpy as np
from tensorflow import keras

def create_sdm(base_dir, model_dir, output_dir, year, month, global_mean, global_std):
    """
    Create a Species Distribution Model (SDM) for a specified month and year.
    
    Parameters:
    base_dir (str): Directory containing environmental data.
    model_dir (str): Directory containing the saved model.
    output_dir (str): Directory to save the output SDM.
    year (int): The year for which to create the SDM.
    month (str): The month for which to create the SDM.
    global_mean (float): Global mean for normalizing the environmental data.
    global_std (float): Global standard deviation for normalizing the environmental data.
    """
    # Load saved model
    model_path = os.path.join(model_dir, f'model_{month}_{year}_l1.h5')
    model = keras.models.load_model(model_path)

    # Load environmental data rasters
    bedrock = rasterio.open(os.path.join(base_dir, year, month.lower(), 'bedrock.tif'))
    distance = rasterio.open(os.path.join(base_dir, year, month.lower(), 'distance_test.tif'))
    slope = rasterio.open(os.path.join(base_dir, year, month.lower(), 'slope.tif'))
    meanchla = rasterio.open(os.path.join(base_dir, year, month.lower(), 'meanchla.tif'))
    meansst = rasterio.open(os.path.join(base_dir, year, month.lower(), 'meansst.tif'))
    salinity = rasterio.open(os.path.join(base_dir, year, month.lower(), 'salinity.tif'))

    grid_rows, grid_cols = slope.shape
    grid_probabilities = np.zeros((grid_rows, grid_cols))

    print(f"Calculating occurrence probabilities for {month} {year}")

    # Prepare the data points for prediction
    env_data_points = []
    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            env_data_point = [
                bedrock.read(1)[grid_row, grid_col],
                distance.read(1)[grid_row, grid_col],
                slope.read(1)[grid_row, grid_col],
                meanchla.read(1)[grid_row, grid_col],
                meansst.read(1)[grid_row, grid_col],
                salinity.read(1)[grid_row, grid_col]
            ]
            env_data_points.append(env_data_point)

    env_data_points = np.array(env_data_points)
    env_data_points_norm = (env_data_points - global_mean) / global_std
    occurrence_probabilities = model.predict(env_data_points_norm)

    # Fill grid_probabilities with the predicted probabilities
    index = 0
    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            grid_probabilities[grid_row, grid_col] = occurrence_probabilities[index][0]
            index += 1

    # Export the grid of occurrence probabilities as a GeoTIFF
    sdm_output = os.path.join(output_dir, f"SDM_{month}_{year}_l1.tif")
    metadata = slope.meta.copy()
    metadata.update({'count': 1, 'dtype': 'float32'})

    with rasterio.open(sdm_output, 'w', **metadata) as dst:
        dst.write(grid_probabilities.astype(np.float32), 1)

    print(f"SDM exported to: {sdm_output}")