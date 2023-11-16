import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#Reading and loading CSV
df = pd.read_csv('DataSet/StressLevelDataset.csv')
print(df.head(), "\n")

#All columns
print("We have ", len(df.columns), " Columns")
print(df.columns, "\n")

#getting basic information about df like  How many students are in the dataset?
df.info()

#he number of unique values for each column
print(df.nunique())

#Data Display
r = 3
c = 7
it = 1
for i in df.columns:
    plt.subplot(r, c, it)
    if df[i].nunique() > 6:
        sns.kdeplot(df[i])
        plt.grid()
    else:
        sns.countplot(x=df[i])
    it += 1
plt.tight_layout()
plt.show()
