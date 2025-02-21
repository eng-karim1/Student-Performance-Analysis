import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

# Display sample data
display(df.sample(5))

# Basic information about the dataset
df.info()

# Statistical summary
display(df.describe())

# Checking for missing values
missing_values = df.isnull().sum()
print("Missing Values per Column:\n", missing_values)

# Handling missing values by replacing them with column means
for column in df.columns:
    if df[column].dtype != 'object':
        df[column].fillna(df[column].mean(), inplace=True)

# Convert categorical data into numerical if necessary
df = pd.get_dummies(df, drop_first=True)

display(df.sample(5))

# Exploratory Data Analysis (EDA)
## Distribution of numerical features
numeric_columns = df.select_dtypes(include=['number']).columns
plt.figure(figsize=(10, 6))
for i, col in enumerate(numeric_columns, 1):
    if i > 6:
        break  # Limiting the number of plots to avoid overcrowding
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

## Relationship between study hours and exam scores
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Hours_Studied"], y=df["Exam_Score"], alpha=0.7)
plt.title("Relationship Between Study Hours and Exam Scores")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.show()

## Correlation analysis
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Correlations')
plt.show()

## Most influential factors for exam performance
correlations = df.corr()['Exam_Score'].sort_values(ascending=False)
print("Most influential factors:\n", correlations)

# Splitting data into features (X) and target variable (y)
y = df['Exam_Score']
X = df.drop(columns=['Exam_Score'])

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
## Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

## Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

## Additional Models
models = {
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = mse
    print(f"{name} MSE: {mse}")

# Model Evaluation
## Linear Regression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f'Linear Regression MAE: {mae_lr}')
print(f'Linear Regression MSE: {mse_lr}')
print(f'Linear Regression RMSE: {rmse_lr}')
print(f'Linear Regression R2 Score: {r2_lr}')

## Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest MAE: {mae_rf}')
print(f'Random Forest MSE: {mse_rf}')
print(f'Random Forest RMSE: {rmse_rf}')
print(f'Random Forest R2 Score: {r2_rf}')

# Error Distribution
errors = y_test - rf_model.predict(X_test)
plt.figure(figsize=(8, 5))
sns.histplot(errors, bins=30, kde=True)
plt.title("Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show()

# Visualizing model performance
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Random Forest: Actual vs Predicted Exam Scores')
plt.show()

plt.figure(figsize=(8,6))
sns.regplot(x=y_test, y=y_pred_lr, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Linear Regression: Actual vs Predicted Exam Scores')
plt.show()
