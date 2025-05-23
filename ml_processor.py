import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MilkQualityML:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.model_path = 'models/milk_quality_model.pkl'
        self.scaler_path = 'models/scaler.pkl'
        self.label_encoder_path = 'models/label_encoder.pkl'
        self.initialize_models()

    def initialize_models(self):
        """Initialize or load the ML models"""
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Check if models exist, if not train them
        if not all(os.path.exists(path) for path in [self.model_path, self.scaler_path, self.label_encoder_path]):
            print("Training new models...")
            self.train_models()
        else:
            print("Loading existing models...")
            self.load_models()

    def train_models(self):
        """Train the machine learning models"""
        # Generate synthetic data for training
        df = self.generate_training_data()
        
        # Preprocess data
        X = df[['pH', 'Turbidity', 'EC', 'Protein', 'SCC']]
        y = df['MilkQuality']
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit label encoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train Random Forest model (best performing model from analysis)
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y_encoded)
        
        # Save models
        self.save_models()

    def generate_training_data(self, n_samples=200000):
        """Generate synthetic training data based on sensor parameters and save to CSV"""
        np.random.seed(42)
        
        # Generate features
        ph_values = np.random.uniform(6.0, 7.5, n_samples)
        turbidity = np.random.uniform(0.5, 10.0, n_samples)
        ec_values = np.random.uniform(4.0, 6.0, n_samples)
        protein_content = np.random.uniform(1.8, 3.8, n_samples)
        
        # SCC generation with some added variability
        w1, w2, C = 1.2, 2.0, 1000
        scc_values = (w1 * ec_values) + (w2 * turbidity) + C
        scc_values += np.random.normal(0, 500000, n_samples)  # larger noise
        scc_values = np.clip(np.abs(scc_values), 1000, 10_000_000)  # clip to realistic limits
        
        # Define quality and action mapping
        action_mapping = {
            'Negative': 'Safe to use',
            'Trace': 'Monitor',
            'Weak_Positive': 'Check the cow',
            'Distinct_Positive': 'Veterinary care',
            'Definite_Positive': 'Reject the milk'
        }
        
        quality_categories = []
        actions = []
        for scc in scc_values:
            if scc <= 200_000:
                label = 'Negative'
            elif scc <= 400_000:
                label = 'Trace'
            elif scc <= 1_200_000:
                label = 'Weak_Positive'
            elif scc <= 5_000_000:
                label = 'Distinct_Positive'
            else:
                label = 'Definite_Positive'
            quality_categories.append(label)
            actions.append(action_mapping[label])
        
        # Build DataFrame
        df = pd.DataFrame({
            'pH': ph_values,
            'Turbidity': turbidity,
            'EC': ec_values,
            'Protein': protein_content,
            'SCC': scc_values,
            'MilkQuality': quality_categories,
            'Action': actions
        })
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/milk_quality_training_data.csv', index=False)
        print(f"Generated {n_samples} samples and saved to data/milk_quality_training_data.csv")
        
        return df

    def save_models(self):
        """Save the trained models"""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.label_encoder, self.label_encoder_path)

    def load_models(self):
        """Load the trained models"""
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoder = joblib.load(self.label_encoder_path)

    def predict_quality(self, sensor_data):
        """
        Predict milk quality based on sensor measurements
        
        Parameters:
        sensor_data (dict): Dictionary containing sensor readings
                           {'ph': float, 'turbidity': float, 'ec': float, 'protein': float, 'scc': float}
        
        Returns:
        dict: Prediction result with quality category and action needed
        """
        # Prepare input data
        input_data = np.array([[
            sensor_data['ph'],
            sensor_data['turbidity'],
            sensor_data['ec'],
            sensor_data['protein'],
            sensor_data['scc']
        ]])
        
        # Scale the input data
        input_data_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_data_scaled)[0]
        quality_category = self.label_encoder.inverse_transform([prediction])[0]
        
        # Determine action based on quality
        action_mapping = {
            'Negative': 'Safe to use',
            'Trace': 'Monitor',
            'Weak_Positive': 'Check the cow',
            'Distinct_Positive': 'Veterinary care',
            'Definite_Positive': 'Reject the milk'
        }
        
        return {
            'milk_quality': quality_category,
            'action_needed': action_mapping.get(quality_category, 'Unknown'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# Create a singleton instance
ml_processor = MilkQualityML() 