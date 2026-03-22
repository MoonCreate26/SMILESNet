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
        return fp_gen.GetFingerprintAsNumPy(mol)
    return None

def fp_to_array(fp):
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

df['fingerprint'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x)).apply(get_fp)

# Set input and output
x = np.stack(df['fingerprint'].values)
y = df['logP'].to_numpy()

print(f"Input: Fingerprint\n{x}\n")
print(f"Output: logP\n{y}\n")

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define Model
model = RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42)

# Train
model.fit(x_train, y_train)

# Predictions
predictions = model.predict(x_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

def try_predict(user_input):
    if user_input == "0":
        print("\nTerminating Program.")
        return
    
    smile = Chem.MolFromSmiles(user_input)

    if smile == None:
        print(f"\nInvalid SMILES: {user_input}")

    else:
        arr = fp_to_array(get_fp(smile))
        print(model.predict([arr]))

    try_predict(input("\nEnter SMILES to analyze or '0' to exit: "))

try_predict(input("\nEnter SMILES to analyze or '0' to exit: "))