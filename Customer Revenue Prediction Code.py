import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set display options for Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.3f}'.format)

# Load the dataset
shop = pd.read_csv("online_shoppers_intention.csv")
shop.head()
# Function to check DataFrame
def explore_dataframe(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe())

# Explore the loaded DataFrame
explore_dataframe(shop, head=2)

# color palette definitoin
colors = ["navy", "lightblue"]

# Graphics
plt.figure(figsize=(8, 6))
sns.countplot(data=shop, x='Revenue', palette=colors)
plt.title('Distribution of Revenue')
plt.show()

# Correlation matrix heatmap
correlation_matrix = shop.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# Select independent variables and the target variable
X = shop[['Administrative_Duration', 'Informational_Duration', 'ProductRelated',
          'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
          'OperatingSystems', 'Browser', 'Region', 'TrafficType',
          'Weekend']]
y = shop['Revenue']

# Encode categorical columns
label_encoder = LabelEncoder()
X['Weekend'] = label_encoder.fit_transform(X['Weekend'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing and Feature Engineering

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Power Transformation to numerical features
power_transformer = PowerTransformer(method='yeo-johnson')
X_train_transformed = power_transformer.fit_transform(X_train_scaled)
X_test_transformed = power_transformer.transform(X_test_scaled)

# Model Selection and Training

# Use GridSearchCV for hyperparameter tuning
# GridSearchCV için parametreleri güncelleyin
# GridSearchCV için parametreleri güncelleyin
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10],
}
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
}
# Set up the Logistic Regression model
# Set up the Logistic Regression model
lr_model = LogisticRegression(C=0.1)
lr_model.fit(X_train_transformed, y_train)

# Set up the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None)
rf_model.fit(X_train_transformed, y_train)

# Set up the CatBoost model
catboost_model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, verbose=0)
catboost_model.fit(X_train_transformed, y_train)

# Set up the XGBoost model
xgboost_model = XGBClassifier()
xgboost_model.fit(X_train_transformed, y_train)

# Test the models
lr_pred = lr_model.predict(X_test_transformed)
rf_pred = rf_model.predict(X_test_transformed)
catboost_pred = catboost_model.predict(X_test_transformed)
xgboost_pred = xgboost_model.predict(X_test_transformed)

# Evaluate model performance
lr_accuracy = accuracy_score(y_test, lr_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

xgboost_accuracy = accuracy_score(y_test, xgboost_pred)

lr_conf_matrix = confusion_matrix(y_test, lr_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_pred)

xgboost_conf_matrix = confusion_matrix(y_test, xgboost_pred)

lr_class_report = classification_report(y_test, lr_pred)
rf_class_report = classification_report(y_test, rf_pred)
xgboost_class_report = classification_report(y_test, xgboost_pred)

print("Logistic Regression Model:")
print("Accuracy:", lr_accuracy)
print("Confusion Matrix:\n", lr_conf_matrix)
print("Classification Report:\n", lr_class_report)

print("Random Forest Model:")
print("Accuracy:", rf_accuracy)
print("Confusion Matrix:\n", rf_conf_matrix)
print("Classification Report:\n", rf_class_report)

print("XGBoost Model:")
print("Accuracy:", xgboost_accuracy)
print("Confusion Matrix:\n", xgboost_conf_matrix)
print("Classification Report:\n", xgboost_class_report)