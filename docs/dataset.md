# Dataset Documentation

## Overview

This dataset contains multimodal sensor recordings from a monitoring system in a car. Data was collected from 15 subjects in highway, city and rural driving scenarios, capturing both motion and respiration signals. A chelt belt was used as ground truth.

## Features


1. **Metadata Columns**:

- `subject_id`: Unique identifier for each subject
- `scenario`: Driving scenario (Highway, city and rural)
- `label`: Binary classification label (1 if a respiratory motion was detected)

2. **Sensor Data Columns**:

- Test and train snippets overlap by 200 and 190 snippets respectively
  - train dataframe shape: 202080 rows × 807 columns
  - test dataframe shape: 2021524 rows × 807 columns

- Each sensor has 201 data points per window, formatted as:

- `piezo_0` to `piezo_200`: raw piezoelectric signal from the seat belt
  - Units: in mV
  
- `accelo_0` to `accelo_200`: de-noised acceloremeter signal from the seat belt
  - Units: in au
  
- `video_belt_0` to `video_belt_200`: Averaged green channel of belt region of interest
  - Units: in au

- `video_chest_0` to `video_chest_200`: Averaged green channel of chest region of interest
  - Units: in au
  

### Dataset Preprocessing

> <http://dx.doi.org/10.1038/s41598-023-47504-y> for comprehensive information

- upsamling to unified frequency of 200 Hz
- Median Filtering and 3. Amplitude normalization
- normalization to interval [-1, 1] was skipped
  - normalization is done after split into train, validation and test dataset
    - in `data_preparation.py` `z_normalize_data()`

### Data Generation

The dataset is generated using the [`preprocessing/generate_dataframe.py`](preprocessing/generate_dataframe.py) script, which:

1. Loads raw data from preprocessed directories
2. Processes data with a window size of 201 points
3. Combines data from all sensors
4. Saves the processed data in Parquet format