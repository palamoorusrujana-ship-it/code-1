import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    """
    A class to predict house prices using machine learning.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
    
    def load_data(self, csv_file):
        """Load house price data from CSV file."""
        try:
            df = pd.read_csv(csv_file)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File '{csv_file}' not found.")
            return None
    
    def preprocess_data(self, df, target_column='price'):
        """
        Preprocess data by handling missing values and encoding categorical variables.
        """
        # Handle missing values
        df = df.dropna()
        
        # Separate features and target
        if target_column not in df.columns:
            print(f"Error: Target column '{target_column}' not found in data.")
            return None, None
        
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        self.feature_names = X.columns.tolist()
        
        print(f"Features: {len(self.feature_names)}")
        print(f"Samples: {len(X)}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model using training data.
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print("\n=== Model Performance ===")
        print(f"Train R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Train RMSE: ${train_rmse:,.2f}")
        print(f"Test RMSE: ${test_rmse:,.2f}")
        print(f"Train MAE: ${train_mae:,.2f}")
        print(f"Test MAE: ${test_mae:,.2f}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
    
    def predict(self, features):
        """
        Predict house price for given features.
        
        Args:
            features: Dictionary or DataFrame with feature values
        
        Returns:
            Predicted price
        """
        if not self.is_trained:
            print("Error: Model is not trained yet.")
            return None
        
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        
        return prediction[0]
    
    def get_feature_importance(self, top_n=10):
        """Get the most important features."""
        if not self.is_trained:
            print("Error: Model is not trained yet.")
            return None
        
        coefficients = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        })
        
        coefficients['abs_coefficient'] = np.abs(coefficients['coefficient'])
        coefficients = coefficients.sort_values('abs_coefficient', ascending=False)
        
        print(f"\nTop {top_n} Important Features:")
        print(coefficients.head(top_n).to_string(index=False))
        
        return coefficients.head(top_n)


def main():
    """Main function to demonstrate house price prediction."""
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Load data (replace 'house_data.csv' with your dataset)
    print("House Price Prediction Model")
    print("=" * 50)
    
    # Example: Load your data
    # df = predictor.load_data('house_data.csv')
    
    # For demonstration, create sample data
    np.random.seed(42)
    n_samples = 500
    
    df = pd.DataFrame({
        'area': np.random.uniform(1000, 5000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'location': np.random.choice(['Downtown', 'Suburb', 'Rural'], n_samples),
        'price': np.random.uniform(100000, 500000, n_samples)
    })
    
    print(f"\nData shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Preprocess
    X, y = predictor.preprocess_data(df, target_column='price')
    
    if X is not None:
        # Train model
        results = predictor.train(X, y)
        
        # Get feature importance
        predictor.get_feature_importance(top_n=5)
        
        # Example prediction
        print("\n=== Example Prediction ===")
        sample_features = {
            'area': 2500,
            'bedrooms': 3,
            'bathrooms': 2,
            'age': 10,
            'location_Rural': 0,
            'location_Suburb': 1
        }
        
        predicted_price = predictor.predict(pd.DataFrame([sample_features]))
        print(f"Predicted Price: ${predicted_price:,.2f}")


if __name__ == "__main__":
    main()
