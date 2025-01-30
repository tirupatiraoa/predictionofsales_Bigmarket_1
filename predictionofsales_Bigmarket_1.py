import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


# Load the dataset
big_mart_data = pd.read_csv('Train.csv')
big_mart_data.head()


# Check the shape and info of the dataset
print("Shape of the dataset:", big_mart_data.shape)
print("\nDataset Info:")
big_mart_data.info()

# Check for missing values
print("Missing values in each column:")
big_mart_data.isnull().sum()

# Fill missing values in 'Item_Weight' with the mean
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)

# Fill missing values in 'Outlet_Size' with the mode based on 'Outlet_Type'
mode_of_outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
missing_value = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[missing_value, 'Outlet_Size'] = big_mart_data.loc[missing_value, 'Outlet_Type'].apply(lambda x: mode_of_outlet_size[x])

# Verify no missing values remain
print("Missing values after imputation:")
big_mart_data.isnull().sum()

# Set the aesthetic style of the plots
sns.set()

# Plot distributions for numerical columns
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(big_mart_data['Item_Weight'], kde=True)
plt.title('Item Weight Distribution')

plt.subplot(2, 2, 2)
sns.histplot(big_mart_data['Item_Visibility'], kde=True)
plt.title('Item Visibility Distribution')

plt.subplot(2, 2, 3)
sns.histplot(big_mart_data['Item_MRP'], kde=True)
plt.title('Item MRP Distribution')

plt.subplot(2, 2, 4)
sns.histplot(big_mart_data['Item_Outlet_Sales'], kde=True)
plt.title('Item Outlet Sales Distribution')

plt.tight_layout()
plt.show()

# Plot count plots for categorical columns
plt.figure(figsize=(20, 15))

plt.subplot(2, 3, 1)
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
plt.title('Outlet Establishment Year')

plt.subplot(2, 3, 2)
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.title('Item Fat Content')

plt.subplot(2, 3, 3)
sns.countplot(x='Item_Type', data=big_mart_data)
plt.xticks(rotation=90)
plt.title('Item Type')

plt.subplot(2, 3, 4)
sns.countplot(x='Outlet_Size', data=big_mart_data)
plt.title('Outlet Size')

plt.subplot(2, 3, 5)
sns.countplot(x='Outlet_Type', data=big_mart_data)
plt.title('Outlet Type')

plt.subplot(2, 3, 6)
sns.countplot(x='Outlet_Location_Type', data=big_mart_data)
plt.title('Outlet Location Type')

plt.tight_layout()
plt.show()


# Standardize 'Item_Fat_Content' values
big_mart_data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)

# Label Encoding for categorical variables
encoder = LabelEncoder()
categorical_columns = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Identifier']

for col in categorical_columns:
    big_mart_data[col] = encoder.fit_transform(big_mart_data[col])

# Display the first few rows after preprocessing
big_mart_data.head()

# Split data into features (X) and target (y)
x = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
y = big_mart_data['Item_Outlet_Sales']

print("Features (X):")
print(x.head())
print("\nTarget (y):")
print(y.head())

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

print("Shape of x_train:", x_train.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Initialize and train the XGBoost Regressor
regression = XGBRegressor()
regression.fit(x_train, y_train)

# Predict on training data
training_data_prediction = regression.predict(x_train)

# Calculate R-squared error
r2_error = metrics.r2_score(y_train, training_data_prediction)
print('R-squared error on training data:', r2_error)

# Predict on testing data
testing_data_prediction = regression.predict(x_test)

# Calculate R-squared error
r2_error = metrics.r2_score(y_test, testing_data_prediction)
print('R-squared error on testing data:', r2_error)