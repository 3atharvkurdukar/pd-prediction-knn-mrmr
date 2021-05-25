
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymrmre import mrmr

# %%
dataset = pd.read_csv('data/parkinsons.csv')
X = dataset.iloc[:, 1:-1]   # input feactures 
y = dataset.iloc[:, -1]     # output

# %%
solution = mrmr.mrmr_ensemble(features=X,targets=y,solution_length=5,solution_count=3)

print(solution)
# https://aka.ms/vs/16/release/vc_redist.x64.exe
