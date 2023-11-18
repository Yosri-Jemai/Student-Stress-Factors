# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading dataset to Notebook via pandas 
df = pd.read_csv("DataSet/StressLevelDataset.csv")
df.head()

# Basic information about df like  How many students are in the dataset
df.info()

#he number of unique values for each column
print(df.nunique())

# Statitical information for numeric columns 
df.describe().round(2)

# Checking for duplicates 
df.duplicated().sum()

# Checking if we have missing values 
df.isna().sum()

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

# Calculate the correlation of every column
correlation_df= df.corr()

# Create a heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(data=correlation_df, annot=True, cmap='coolwarm')
plt.title("What tends to a higher stress level?", fontweight='bold')
plt.show()
