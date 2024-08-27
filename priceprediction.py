# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data creation
# Assuming we have some data about houses with their prices, square footage, number of bedrooms, and bathrooms
data = pd.read_csv('train.csv')

# Convert data into a pandas DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print("Dataset:")
print(df.head())

# Features and target variable
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]  # Features
y = df['Price']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared value (Coefficient of Determination)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the coefficients of the model
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Square Footage Coefficient: {model.coef_[0]}")
print(f"Bedrooms Coefficient: {model.coef_[1]}")
print(f"Bathrooms Coefficient: {model.coef_[2]}")

# Print the evaluation metrics
print("\nEvaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Display predictions vs actual prices
print("\nPredicted vs Actual Prices:")
comparison = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
print(comparison)
