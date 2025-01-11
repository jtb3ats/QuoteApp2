import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from flask import Flask, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Load the dataset (assumes CSV file with job details and quote prices)
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Train the Random Forest model
def train_model(data):
    # Assume the data has features: area, labor_hours, material_cost, etc., and target: price
    X = data[['area', 'labor_hours', 'material_cost']]  # Features
    y = data['price']  # Target variable
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    # Save the model to a file
    joblib.dump(model, 'landscaping_model.pkl')

# Load the pre-trained model
def load_model():
    return joblib.load('landscaping_model.pkl')

# Route to train the model (can be triggered manually)
@app.route('/train', methods=['POST'])
def train():
    try:
        # Load data from uploaded CSV file
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Train the model on the uploaded data
        train_model(df)
        
        return jsonify({"message": "Model training complete!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to predict a new quote
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data (e.g., area, labor_hours, material_cost)
        input_data = request.get_json()
        model = load_model()
        
        # Prepare input data
        features = np.array([[
            input_data['area'],
            input_data['labor_hours'],
            input_data['material_cost']
        ]])
        
        # Make a prediction
        predicted_price = model.predict(features)[0]
        
        return jsonify({"predicted_price": predicted_price}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to check model performance (e.g., accuracy, error)
@app.route('/model_performance', methods=['GET'])
def model_performance():
    try:
        model = load_model()
        # Load a sample data set for evaluation (you can adjust this)
        df = load_data('sample_test_data.csv')
        X = df[['area', 'labor_hours', 'material_cost']]
        y = df['price']
        
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        return jsonify({"Mean Squared Error": mse}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
