import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data from the CSV file.  This is really just an example of something you may come across from somebody when predicting pricing.  
# This needs to have a column labelled 'product' and another labeled 'price'.  It's a really really simple case here, but this can be expanded.
data = pd.read_csv('prices.csv')

# Convert the 'product' column to numerical values using one-hot encoding to represent categorical variables to improve accuracy. 
data = pd.get_dummies(data, columns=['product'])
print(data)

# Split the data into training and testing sets.
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a sklearn linear model regression model.
model = LinearRegression()

# Train the model on the training data.
model.fit(X_train, y_train)

# Make predictions on the testing data.
predictions = model.predict(X_test)

# Evaluate the model's performance by measuring the average of the squares of the errors. 
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Use the model to make predictions on new data.
new_product = pd.DataFrame({'product_A': [1], 'product_B': [0]})  # Replace with new product data.
prediction = model.predict(new_product)
print(f'Predicted price: {prediction[0]}')