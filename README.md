# SMILESNet
A machine learning model trained on Kaggle dataset to predict simple organic compound's solubility in water/oil.

# Context
SMILES is a string representation of chemical compounds. The purpose of this project is to experiment around with ML implementation in the field of cheminformatics. CompoundAnalysis.py takes in string input of SMILES, converts it to bit vector, then feeds it to random forest algorithm from scikit-learn library, to predict logP, the type of liquid in which the substance dissolves in - important in researching how the drug interacts with the body. Uses small organic compound dataset from Kaggle. The project helped in learning the chemistry library in Python, while getting more used to standard ML library like scikit-learn.