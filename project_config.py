'''
PROJECT CONFIGURATION FILE
This file contains the configuration variables for the project. The variables are used 
in the other scripts to define the paths to the data and results directories. The variables 
are also used to set the random seed for reproducibility.
'''

# import libraries
import os
from pathlib import Path

# check if on O2 or not
home_variable = os.getenv('HOME')
on_remote = (home_variable == "/home/an252")

# define base project directory based on whether on O2 or not
if on_remote:
    PROJECT_DIR = Path('/n/data1/hms/dbmi/zitnik/lab/users/an252/NeuroKG/neuroKG')
else:
    PROJECT_DIR = Path('/Users/an583/Library/CloudStorage/OneDrive-Personal/Research/Zitnik Lab/NeuroKG/neuroKG')

# define project configuration variables
DATA_DIR = PROJECT_DIR / 'Data'
KG_DIR = DATA_DIR / 'NeuroKG' / '3_harmonize_KG'
RESULTS_DIR = PROJECT_DIR / 'Results'
SEED = 42

# CZ CELLxGENE Census dataset variables
CELLXGENE_DATASET = Path('/n/data1/hms/dbmi/zitnik/lab/datasets/2023-05-CELLxGENE')
CELLXGENE_DIR = DATA_DIR / 'CELLxGENE'