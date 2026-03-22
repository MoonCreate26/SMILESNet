import numpy as np;
import kagglehub
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Load the dataset
df = pd.read_csv('SMILES_Big_Data_Set.csv')

# Convert SMILES -> mol -> fingerprints
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_fp(mol):
    if mol:
        return fp_gen.GetFingerprint(mol)
    return None

df['fingerprint'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x)).apply(get_fp)

x = df['fingerprint']


print(x.iloc[0])