# Healthcare in Motion: Development of an Integrated Respiratory Rate Detection System for Drivers Enabled with Signal Fusion

This repository contains the source code for the IEEE EMBC 2025 conference paper `Healthcare in Motion: Development of an Integrated Respiratory Rate Detection System for Drivers Enabled with Signal Fusion`

## Project Structure

```sh
├── container                       # Stores Singularity file and container
├── data/                           # Stores input data
├── docs/                           # Stores documentation
├── out/                            # Stores output data
├── src/                            # Contains Source Code
│   ├── models/                     # Contains model architectures
│   ├── preprocessing/              # scripts are used to generate the dataframes (not necessary if you downloaded the dataset already)
│   └── utils/                      # Utility functions for data processing, visualization, and logging
├── jobs.csv                        # CSV file of learning rates and weight decays for `run_jobs.py`
├── justfile                        # Runs the singularity commands
├── LICENSE                         # License Agreement
├── main.py                         # Main script for training and evaluating models
└── README.md                       # Documentation
└── requirements.txt                # required packages
```

## Setup

### Prerequisites

- **Download dataset as parquet dataframes from <https://leopard.tu-braunschweig.de/receive/dbbs_mods_00078415>**
- Supported Python versions: 3.10 - 3.12
- TensorFlow 2.17
- Additional libraries: check `requirements.txt`

### Installation

1. Create virtual environment: `python -m venv venv`
2. Activate virtual environment: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## Usage

### Configuration

Modify `src/utils/config.py` to adjust settings like model type, optimizer, batch size, and epochs.
Add models in `src/utils/tf_utils.py` to `get_model()`.

### Training and Evaluation

Run the main script to train and evaluate models:
```bash
python main.py
```

Run the main script consecutively with learning rates and weight decay parameters defined in `jobs.csv`:
```bash
python src/utils/run_jobs.py jobs.csv
```


### Additional Flags

| Flags           | Description                            |
|--------------------------|----------------------------------------|
| --train_one_subject_only | use only one subject instead of all 15 |
| --train_subset           | use only 3 batches of data per subject |



## Alternatively: use the Singularity Image instead

> Install [Singularity](https://docs.sylabs.io/guides/latest/user-guide/)
> Install [Just](https://github.com/casey/just)

- Build image: `just build_sing`
  - one some systems (e.g. Fedora) the size of the the /tmp dir is limited.
    Use the following command to increase the size of the /tmp dir:
    ```bash
    sudo mount -o remount,size=16G,noatime /tmp
    ```
  - or download image from <https://leopard.tu-braunschweig.de/receive/dbbs_mods_00078415>
- Check python version: `just check_package_versions`
- Check for GPU: `just check_gpu`
- Run main.py: `just main {{ARGS}}`
  - e.g. `just main --train_one_subject_only` use only one subject instead of all 15
- Run jobs: `just run_jobs {{ARGS}}`
  - e.g. `just run_jobs jobs.csv --train_one_subject_only --train_subset` use only one subject and only 3 batches
  - e.g. `just run_jobs jobs.csv` train complete LOSOCV pipeline with all data


## Hardware & Software used

CPU: Intel(R) Core(TM) i9-14900K
GPU: NVIDIA RTX 4500 Ada Generation, 24 GB
CUDA: 12.8
OS: Ubuntu 24.04
