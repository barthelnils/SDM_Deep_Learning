
import rasterio
import numpy as np
import pandas as pd

def load_data(base_dir, year, month):
    """
    Load environmental rasters and occurrence points for a specific month and year.

    Parameters:
    base_dir (str): The base directory where the raster and CSV files are located.
    year (int): The year for which data is to be loaded.
    month (int): The month for which data is to be loaded.

    Returns:
    env_data (numpy array): Array of environmental data points for the occurrence locations.
    labels (numpy array): Array of labels corresponding to each occurrence point (species).
    """
    
    # Load environmental rasters
    bedrock = rasterio.open(f'{base_dir}/{year}/{month}/bedrock.tif')
    distance = rasterio.open(f'{base_dir}/{year}/{month}/distance.tif')
    slope = rasterio.open(f'{base_dir}/{year}/{month}/slope.tif')
    meanchla = rasterio.open(f'{base_dir}/{year}/{month}/meanchla.tif')
    meansst = rasterio.open(f'{base_dir}/{year}/{month}/meansst.tif')
    salinity = rasterio.open(f'{base_dir}/{year}/{month}/salinity.tif')

    # Load occurrence points saved in csv-file
    occurrence_points_path = f'{base_dir}/{year}/{month}/hw_{year}_{month}_filtered.csv'
    occurrence_points = pd.read_csv(occurrence_points_path)

    # Create empty arrays to store environmental data and labels
    env_data = []
    labels = []
    for i, row in occurrence_points.iterrows():
        try:
            row, col = bedrock.index(row['longitude'], row['latitude'])
            env_data_point = [
                bedrock.read(1)[row, col],
                distance.read(1)[row, col],
                slope.read(1)[row, col],
                meanchla.read(1)[row, col],
                meansst.read(1)[row, col],
                salinity.read(1)[row, col]
            ]
            env_data.append(env_data_point)
            labels.append(occurrence_points.loc[i, 'species'])
        except ValueError:
            print(f"Skipping occurrence point {i+1}")

    env_data = np.array(env_data)
    labels = np.array(labels)
    labels = labels.astype('float32')

    return env_data, labels
