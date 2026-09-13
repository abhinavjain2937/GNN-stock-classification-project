import pandas as pd
import os 
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR/'data'/'processed'/'nifty50_features.csv'

if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print("----- Success! Dataset loaded perfectly.")
    print(f"Dataset Shape: {data.shape}")
    print(data.head(10))
else:
    print(f"---  Error: Could not find the file at:\n{file_path}")






# src_dir = os.path.dirname(os.path.abspath(__file__))

# # 2. Go up one level to the project root, then down into the data folder
# project_root = os.path.dirname(src_dir)
# file_path = os.path.join(project_root, "data", "processed", "nifty50_features.csv")


# if os.path.exists(file_path):
#     data = pd.read_csv(file_path)
#     print("----- Success! Dataset loaded perfectly.")
#     print(f"Dataset Shape: {data.shape}")
#     print(data.head(10))
# else:
#     print(f"---  Error: Could not find the file at:\n{file_path}")

