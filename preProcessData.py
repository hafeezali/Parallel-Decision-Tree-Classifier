
# coding: utf-8

# In[23]:

import csv
import pandas as pd
import numpy as np


# In[24]:

if __name__ == "__main__":
    fileName = "nursery.data.txt"
    df = pd.read_csv(fileName)
    for col in df.columns:
        df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes + 1
    file_name = "nursery-data.int.txt"
    df = df.sample(frac=0.6).reset_index(drop=True)
    df.to_csv(file_name, sep=',',header=None)
    
    

