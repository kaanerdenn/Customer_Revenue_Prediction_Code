# Customer_Revenue_Prediction_Code
# Online Shopper's Intention Prediction

This repository contains a Python script for predicting online shopper's intention using various machine learning models. The code includes data preprocessing, feature engineering, model selection, training, and evaluation.

## Dataset

The dataset used in this project is loaded from the "online_shoppers_intention.csv" file. It contains information about online shopping sessions, including various numerical and categorical features. The target variable is "Revenue," which indicates whether a visitor made a purchase (1) or not (0).

### Data Exploration

The script begins with data exploration using the `explore_dataframe` function to provide an overview of the dataset's shape, data types, head, missing values, and summary statistics.

## Data Visualization

### Distribution of Revenue

A countplot is created to visualize the distribution of the target variable "Revenue." The plot helps us understand the balance between successful and unsuccessful online shopping sessions.

### Correlation Matrix Heatmap

A correlation matrix heatmap is generated to visualize the pairwise correlations between numerical features. This helps identify potential relationships between features.

## Preprocessing and Feature Engineering

### Encoding Categorical Columns

The "Weekend" column, a categorical feature, is encoded using label encoding to convert it into numerical form.

### Splitting the Data

The dataset is split into training and test sets using a 80-20 split ratio. This allows for model training on one part and evaluation on the other.

### Standardization and Power Transformation

Numerical features are standardized using the StandardScaler and power-transformed using the Yeo-Johnson method to improve their distribution.

## Model Selection and Training

Several machine learning models are selected and trained on the preprocessed data:

- **Logistic Regression**: A logistic regression model is trained with hyperparameter C=0.1.
- **Random Forest**: A random forest classifier is trained with 100 estimators and no specified maximum depth.
- **CatBoost**: A CatBoost classifier is trained with 1000 iterations, depth=6, and learning rate=0.1.
- **XGBoost**: An XGBoost classifier is trained with default hyperparameters.

## Model Evaluation

The trained models are evaluated on the test data, and the following metrics are calculated:

- **Accuracy**: The accuracy score measures the overall model performance.
- **Confusion Matrix**: A confusion matrix is generated to visualize true positives, true negatives, false positives, and false negatives.
- **Classification Report**: A classification report provides precision, recall, F1-score, and support for each class.

## Results

The results of the model evaluation are displayed for each model, including accuracy, confusion matrix, and classification report.
