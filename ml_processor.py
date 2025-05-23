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
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Save models
        self.save_models()

    def generate_training_data(self, n_samples=200000):
        """Generate synthetic training data based on sensor parameters and save to CSV"""
        np.random.seed(42)
        
        # Generate features with wider ranges to ensure all quality categories
        ph_values = np.random.uniform(5.5, 7.5, n_samples)  # Expanded pH range
        turbidity = np.random.uniform(0.5, 30.0, n_samples)  # Expanded turbidity range
        ec_values = np.random.uniform(3.5, 7.0, n_samples)   # Expanded EC range
        protein_content = np.random.uniform(1.5, 4.0, n_samples)  # Expanded protein range
        
        # Generate SCC values independently - not as a function of other parameters
        scc_distribution = np.random.choice(['low', 'medium', 'high', 'very_high', 'extreme'], 
                                          size=n_samples, 
                                          p=[0.4, 0.3, 0.15, 0.1, 0.05])  # Adjusted probabilities
        
        scc_values = np.zeros(n_samples)
        # Create a mix of safe and unsafe milk with realistic SCC ranges
        for i, dist in enumerate(scc_distribution):
            if dist == 'low':
                scc_values[i] = np.random.uniform(10000, 200000)
            elif dist == 'medium':
                scc_values[i] = np.random.uniform(200001, 400000)
            elif dist == 'high':
                scc_values[i] = np.random.uniform(400001, 1200000)
            elif dist == 'very_high':
                scc_values[i] = np.random.uniform(1200001, 5000000)
            else:  # extreme
                scc_values[i] = np.random.uniform(5000001, 10000000)
        
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
        
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate multiple machine learning models"""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        best_accuracy = 0
        best_model_name = None
        
        print("\nModel Evaluation Results:")
        print("-" * 40)
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n{name} Accuracy: {accuracy:.4f}")
            print(classification_report(y_test, y_pred, 
                                      target_names=self.label_encoder.classes_,
                                      zero_division=0))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                self.model = model
        
        print(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        print("-" * 40)

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