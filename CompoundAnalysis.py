import numpy as np;
import kagglehub

import pandas as pd
import os

# List files to confirm path
print(os.listdir('.'))

# Load the dataset
df = pd.read_csv('your-file-name.csv')

print(df.head())