import os
import numpy as np
import rasterio
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import yaml

def project_model(config, year):
    """
    Project the model on environmental data for a specific year.
    
    Parameters:
    config (dict): Configuration dictionary containing model and data paths.
    year (int): The year for which to project the model.
    """
    model_path = os.path.join(config['model_dir'], config['final_model_name'])
    model = keras.models.load_model(model_path)

    months = config['projection_months']
    base_dir = config['base_dir']
    output_dir = config['output_dir']

    for month in months:
        bedrock = rasterio.open(f'{base_dir}/{year}/{month}/bedrock.tif')
        distance = rasterio.open(f'{base_dir}/{year}/{month}/distance.tif')
        slope = rasterio.open(f'{base_dir}/{year}/{month}/slope.tif')
        meanchla = rasterio.open(f'{base_dir}/{year}/{month}/meanchla.tif')
        meansst = rasterio.open(f'{base_dir}/{year}/{month}/meansst.tif')
        salinity = rasterio.open(f'{base_dir}/{year}/{month}/salinity.tif')

        env_data = []
        for row in range(bedrock.height):
            for col in range(bedrock.width):
                try:
                    env_data_point = [
                        bedrock.read(1)[row, col],
                        distance.read(1)[row, col],
                        slope.read(1)[row, col],
                        meanchla.read(1)[row, col],
                        meansst.read(1)[row, col],
                        salinity.read(1)[row, col]
                    ]
                    env_data.append(env_data_point)
                except ValueError:
                    continue

        env_data = np.array(env_data)
        env_data = np.delete(env_data, np.argwhere(np.isnan(env_data))[:, 0], axis=0)

        mean = np.mean(env_data, axis=0)
        std = np.std(env_data, axis=0)

        grid_rows, grid_cols = slope.shape
        grid_probabilities = np.zeros((grid_rows, grid_cols))
        batch_size = 64
        for i in range(0, grid_rows, batch_size):
            for j in range(0, grid_cols, batch_size):
                env_data_batch = np.array([
                    [
                        bedrock.read(1)[x, y],
                        distance.read(1)[x, y],
                        slope.read(1)[x, y],
                        meanchla.read(1)[x, y],
                        meansst.read(1)[x, y],
                        salinity.read(1)[x, y]
                    ]
                    for x in range(i, min(i + batch_size, grid_rows))
                    for y in range(j, min(j + batch_size, grid_cols))
                ])
                env_data_batch_norm = (env_data_batch - mean) / std
                occurrence_probabilities_batch = model.predict(env_data_batch_norm)
                grid_probabilities[i:i + batch_size, j:j + batch_size] = occurrence_probabilities_batch.reshape(
                    min(batch_size, grid_rows - i), min(batch_size, grid_cols - j)
                )

        scaler = MinMaxScaler(feature_range=(0, 1))
        grid_probabilities_rescaled = scaler.fit_transform(grid_probabilities.reshape(-1, 1)).reshape(grid_rows, grid_cols)

        output_file = os.path.join(output_dir, f"SDM_{month}_{year}_l1_proj.tif")
        metadata = slope.meta.copy()
        metadata.update({'count': 1, 'dtype': 'float32'})
        with rasterio.open(output_file, 'w', **metadata) as dst:
            dst.write(grid_probabilities_rescaled.astype(np.float32), 1)

        print(f"Occurrence probabilities grid exported to: {output_file}")

# Example usage
if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    project_model(config, config['projection_year'])
