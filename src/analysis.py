import pickle
import numpy as np
import matplotlib.pyplot as plt

PICKLE_PATH = "pickle_jar/result.pkl"

with open(PICKLE_PATH,"rb") as handle:
    result = pickle.load(handle)

atoms = result["atoms"]
Afield = result["field"]


