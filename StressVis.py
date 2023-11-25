# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading dataset to Notebook via pandas 
data = pd.read_csv("DataSet/StressLevelDataset.csv")
data.head()

# Basic information about df like  How many students are in the dataset
data.info()

#he number of unique values for each column
print(data.nunique())

# Statitical information for numeric columns 
data.describe().round(2)

# Checking for duplicates 
data.duplicated().sum()

# Checking if we have missing values 
data.isna().sum()

#Data Display
r = 3
c = 7
it = 1
for i in data.columns:
    plt.subplot(r, c, it)
    if data[i].nunique() > 6:
        sns.kdeplot(data[i])
        plt.grid()
    else:
        sns.countplot(x=data[i])
    it += 1
plt.tight_layout()
plt.show()

# Calculate the correlation of every column
correlation_df= data.corr()

# Create a heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(data=correlation_df, annot=True, cmap='coolwarm')
plt.title("What tends to a higher stress level?", fontweight='bold')
plt.show()
print("data --------------")


#top=0.984


data.plot(kind="box",subplots=True,layout=(6,2),figsize=(15,20));
