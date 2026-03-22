# Main imports
import kagglehub
import numpy as np;
import pandas as pd
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# RDKit
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs


# Load the dataset
df = pd.read_csv('SMILES_Big_Data_Set.csv')

# Convert SMILES -> mol -> Bit Vector -> Np Array
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_fp(mol):
    if mol:
        return fp_gen.GetFingerprint(mol)
    return None

def fp_to_array(fp):
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

df['fingerprint'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x)).apply(get_fp).apply(lambda x: fp_to_array(x) if x else None)

# Set input and output
x = df['fingerprint']
y = df['logP']

print(f"Input: Fingerprint\n{x}\n")
print(f"Output: logP\n{y}")