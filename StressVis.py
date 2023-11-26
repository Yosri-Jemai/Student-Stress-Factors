# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading dataset to Notebook via pandas
ds=pd.read_csv("DataSet/StressLevelDataset.csv")
ds.head()

# Basic information about df like  How many students are in the dataset
ds.info()

#The number of unique values for each column
print(ds.nunique())

#Data Display
r=5
c=5
# r=3
# c=7
it=1
for i in ds.columns:
    plt.subplot(r,c,it)
    if ds[i].nunique()>6:
        sns.kdeplot(ds[i]) #Density
        plt.grid()
    else:
        sns.countplot(x=ds[i]) #count
    it+=1
plt.tight_layout()
plt.show()

# Statistical information for numeric columns
ds.describe().round(2)

ds.duplicated().sum()

ds.isna().sum()

# Calculate the correlation of every column
correlation_data= ds.corr()

# Create heatmap using seaborn library
plt.figure(figsize=(14, 10))
sns.heatmap(data=correlation_data, annot=True, cmap='coolwarm')
plt.title("What tends to a higher stress level?", fontweight='bold')
plt.show()

# Filtering the dataset with only physiological factors
physiological_factors = ds[['headache', 'blood_pressure', 'sleep_quality', 'breathing_problem', 'stress_level']]

# Calculating correlation
correlation_physiological = physiological_factors.corr()

# Creating heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data=correlation_physiological, annot=True, cmap='coolwarm')
plt.title('Physiological factors vs Stress levels')
plt.show()

students_health_issue = ds[ds['mental_health_history']==1]
S_count = len(students_health_issue)
print ('Number of students that have reported a health issue is: ' ,S_count)

print('The porcentahe of students that have reported a health issue is: ',round((S_count*100)/1100))

# Visualizing anxiety levels in students
plt.figure(figsize=(8,6))
#data=df
ax = sns.histplot(data=ds, x='anxiety_level', bins=20, color='lightcoral')
plt.title('Anxiety Levels in students')
plt.xlabel('Anxity Level')
plt.ylabel('Number of Students')
plt.show()
# Calculating the average of anxiety in students
average_anxiety = ds['anxiety_level'].mean().round(2)
print('Average anxiety of students in this dataset is ', average_anxiety)

# Defining the correlation between two columns
correlation= ds['study_load'].corr(ds['stress_level']).round(2)
print('The correlation between headaches and higher stress level is ', correlation)

# Calculating average self-steem level of the dataset
average_selfsteem = ds['self_esteem'].mean().round(2)
print('The averrage of self-steem in the dataset is: ', average_selfsteem)

# Counting number of students that have less than the average of self-steem
students_below_average = ds[ds['self_esteem'] < average_selfsteem]['self_esteem'].count()
print('Number of students that have less than the average level of self-steem is: ', students_below_average)

# Visualizing
plt.figure(figsize=(8, 6))
plt.bar(['Students Below Average', 'Students Greater Average'], [students_below_average, len(ds)-students_below_average],
        color=['lightcoral', 'gray'])
plt.title('Self-Esteem Levels in Students')
plt.ylabel('Number of Stundents')
plt.show()

# Calculating number of students that experience depression
students_with_depression = ds[ds['depression']>0].shape[0]
print('Number of students of the dataset with depression is: ', students_with_depression)

# Calculating the percentage
percentage_depression = round(students_with_depression / len(ds) * 100,2)
print('Percentage of students with depression is: ', percentage_depression)

# Visualizong with pie chart
labels = ['Students with Depression', 'Students with no Depression']
sizes = [students_with_depression, 1100 - students_with_depression]
colors = ['lightcoral', 'gray']
explode = (0.1, 0)

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors)
plt.title('Percentage of Students with Depression')
plt.show()

# Calculating number of students have experienced bullying
students_bullying = ds[ds['bullying']>0].shape[0]
print('Number of Students with depression is: ', students_bullying )

# Calculating the percentage of students that have experience bullying
percentage_bullying = round(students_bullying / len(ds) * 100,2)
print('Percentage of students that have experience bullying is: ', percentage_bullying)

# Visualizing bullyig percentage with a pie chart
labels = ['Students bullied', 'Students not bullied']
sizes = [students_bullying, 1100 - students_bullying]
colors = ['lightcoral', 'gray']
explode = (0.1, 0)

plt.figure(figsize=(10,8))
plt.pie(sizes,labels=labels, colors=colors, explode=explode)
plt.title('Students with Bullying experience')
plt.show()

#Outlier detection using boxplot (values that deviate from the overall pattern of the data and may be unusually high or low)
ds.plot(kind="box",subplots=True,layout=(7,3),figsize=(16,28));
#box plot can visually represent the distribution of the data and highlight potential outliers.


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

X = ds[['self_esteem']]
y = ds[['stress_level']]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Size of the training set (X_train, y_train):", X_train.shape, y_train.shape)
print("Size of the test set (X_test, y_test):", X_test.shape, y_test.shape)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test) # m * X_train + c

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the results
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red', label='Linear Regression')
plt.xlabel('Self Esteem')
plt.ylabel('Stress Level')
plt.legend()
plt.title('Linear Regression of Stress Level Depending on Self Esteem')
plt.show()

