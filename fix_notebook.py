import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.8.10"
    }
}

nb['cells'] = []
nb['cells'].append(nbf.v4.new_markdown_cell("# Milk Quality Analysis and Machine Learning Pipeline\n\nThis notebook demonstrates the complete pipeline for milk quality analysis, including data generation with realistic variations, exploratory data analysis, model training and evaluation with multiple algorithms."))

imports_cell = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
"""
nb['cells'].append(nbf.v4.new_code_cell(imports_cell))

nb['cells'].append(nbf.v4.new_markdown_cell("## 1. Data Generation with Realistic Variations\n\nGenerate synthetic milk quality data with sufficient randomness to ensure realistic model performance."))

data_gen_cell = """
def generate_training_data(n_samples=50000):
    '''Generate synthetic training data with realistic variations'''
    np.random.seed(42)
    
    ph_values = np.random.uniform(5.5, 7.5, n_samples)  # Expanded pH range
    turbidity = np.random.uniform(0.5, 30.0, n_samples)  # Expanded turbidity range
    ec_values = np.random.uniform(3.5, 7.0, n_samples)   # Expanded EC range
    protein_content = np.random.uniform(1.5, 4.0, n_samples)  # Expanded protein range
    
    scc_distribution = np.random.choice(['low', 'medium', 'high', 'very_high', 'extreme'], 
                                      size=n_samples, 
                                      p=[0.4, 0.3, 0.15, 0.1, 0.05])  # Adjusted probabilities
    
    scc_values = np.zeros(n_samples)
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
    
    action_mapping = {
        'Negative': 'Safe to use',
        'Trace': 'Monitor',
        'Weak_Positive': 'Check the cow',
        'Distinct_Positive': 'Veterinary care',
        'Definite_Positive': 'Reject the milk'
    }
    
    quality_categories = []
    actions = []
    
    for i in range(n_samples):
        scc = scc_values[i]
        ph = ph_values[i]
        turb = turbidity[i]
        ec = ec_values[i]
        
        if np.random.random() < 0.2:
            label = np.random.choice(['Negative', 'Trace', 'Weak_Positive', 'Distinct_Positive', 'Definite_Positive'])
        else:
            if scc <= 200_000:
                if (ph < 6.0 or ph > 7.0 or turb > 5.0 or ec > 5.5) and np.random.random() < 0.3:
                    label = 'Trace'
                else:
                    label = 'Negative'
            elif scc <= 400_000:
                rand = np.random.random()
                if rand < 0.3:
                    label = 'Negative'
                elif rand < 0.6:
                    label = 'Weak_Positive'
                else:
                    label = 'Trace'
            elif scc <= 1_200_000:
                rand = np.random.random()
                if rand < 0.3:
                    label = 'Trace'
                elif rand < 0.6:
                    label = 'Distinct_Positive'
                else:
                    label = 'Weak_Positive'
            elif scc <= 5_000_000:
                rand = np.random.random()
                if rand < 0.3:
                    label = 'Weak_Positive'
                elif rand < 0.6:
                    label = 'Definite_Positive'
                else:
                    label = 'Distinct_Positive'
            else:
                if np.random.random() < 0.3:
                    label = 'Distinct_Positive'
                else:
                    label = 'Definite_Positive'
        
        quality_categories.append(label)
        actions.append(action_mapping[label])
    
    df = pd.DataFrame({
        'pH': ph_values,
        'Turbidity': turbidity,
        'EC': ec_values,
        'Protein': protein_content,
        'SCC': scc_values,
        'MilkQuality': quality_categories,
        'Action': actions
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/milk_quality_training_data.csv', index=False)
    print(f"Generated {n_samples} samples and saved to data/milk_quality_training_data.csv")
    
    return df
"""
nb['cells'].append(nbf.v4.new_code_cell(data_gen_cell))

nb['cells'].append(nbf.v4.new_code_cell("# Generate the data\ndf = generate_training_data()\ndf.head()"))

nb['cells'].append(nbf.v4.new_markdown_cell("## 2. Exploratory Data Analysis\n\nAnalyze the distribution of milk quality categories and relationships between features."))

nb['cells'].append(nbf.v4.new_code_cell("""# Display quality distribution
quality_counts = df['MilkQuality'].value_counts()
print("Milk Quality Distribution:")
print(quality_counts)
print("\\nPercentage Distribution:")
print(quality_counts / len(df) * 100)"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Plot quality distribution
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='MilkQuality', order=df['MilkQuality'].value_counts().index, palette='viridis')
plt.title('Milk Quality Distribution', fontsize=14)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Quality Category', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Sample data for visualization (using a smaller subset for better visualization)
df_sample = df.sample(1000, random_state=42)

sns.pairplot(df_sample, hue='MilkQuality', vars=['pH', 'Turbidity', 'EC', 'Protein', 'SCC'])
plt.suptitle('Pairplot of Sampled Milk Data', y=1.02, fontsize=16)
plt.show()"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Create boxplots to visualize feature distributions by quality category
features = ['pH', 'Turbidity', 'EC', 'Protein', 'SCC']

fig, axes = plt.subplots(len(features), 1, figsize=(12, 15))
for i, feature in enumerate(features):
    sns.boxplot(x='MilkQuality', y=feature, data=df_sample, ax=axes[i])
    axes[i].set_title(f'{feature} by Milk Quality', fontsize=12)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()"""))

nb['cells'].append(nbf.v4.new_markdown_cell("## 3. Data Preprocessing\n\nPrepare the data for model training."))

nb['cells'].append(nbf.v4.new_code_cell("""# Extract features and target
X = df[['pH', 'Turbidity', 'EC', 'Protein', 'SCC']]
y = df['MilkQuality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Label Encoding Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {i}")"""))

nb['cells'].append(nbf.v4.new_markdown_cell("## 4. Train/Test Split (80/20)\n\nSplit the data for training and evaluation."))

nb['cells'].append(nbf.v4.new_code_cell("""# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_scaled):.1%})")
print(f"Testing set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(X_scaled):.1%})")"""))

nb['cells'].append(nbf.v4.new_markdown_cell("## 5. Model Training and Evaluation\n\nTrain and evaluate multiple machine learning models."))

nb['cells'].append(nbf.v4.new_code_cell("""# Define models to evaluate
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

results = {}
best_accuracy = 0
best_model = None
best_model_name = None

for name, model in models.items():
    print(f"\\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")"""))

nb['cells'].append(nbf.v4.new_markdown_cell("## 6. Model Analysis\n\nAnalyze the best performing model."))

nb['cells'].append(nbf.v4.new_code_cell("""# Generate confusion matrix for the best model
y_pred_best = best_model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', 
            xticklabels=list(label_encoder.classes_), 
            yticklabels=list(label_encoder.classes_), 
            cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.show()"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Feature importance analysis (if available)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    features = ['pH', 'Turbidity', 'EC', 'Protein', 'SCC']
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("Feature Importance:")
    for index, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
else:
    print(f"The {best_model_name} model does not provide feature importance information.")"""))

nb['cells'].append(nbf.v4.new_markdown_cell("## 7. Model Comparison\n\nCompare the performance of all models."))

nb['cells'].append(nbf.v4.new_code_cell("""# Plot model accuracy comparison
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
accuracies = list(results.values())

bars = plt.bar(model_names, accuracies, color=['#4361ee', '#3a0ca3', '#f72585', '#4cc9f0'])
plt.title('Model Accuracy Comparison', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.0)  # Full scale to show realistic accuracy

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Cross-validation results
cv_results = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
    cv_results[name] = cv_scores
    print(f"{name} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

plt.figure(figsize=(12, 6))
plt.boxplot([cv_results[name] for name in model_names], labels=model_names)
plt.title('Cross-Validation Results', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()"""))

nb['cells'].append(nbf.v4.new_markdown_cell("## 8. Model Saving\n\nSave the best model for future use."))

nb['cells'].append(nbf.v4.new_code_cell("""# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

joblib.dump(best_model, 'models/milk_quality_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')

print(f"Best model ({best_model_name}) and preprocessing components saved successfully.")"""))

nb['cells'].append(nbf.v4.new_markdown_cell("## 9. Prediction Examples\n\nDemonstrate how to use the model for predictions."))

nb['cells'].append(nbf.v4.new_code_cell("""# Example milk readings
example_readings = [
    {'ph': 6.8, 'turbidity': 2.5, 'ec': 4.5, 'protein': 3.2, 'scc': 150000},  # Should be Negative/Healthy
    {'ph': 6.3, 'turbidity': 6.0, 'ec': 5.2, 'protein': 2.9, 'scc': 350000},  # Should be Trace/Monitor
    {'ph': 5.8, 'turbidity': 15.0, 'ec': 6.5, 'protein': 2.2, 'scc': 3000000}  # Should be Distinct_Positive/Problematic
]

action_mapping = {
    'Negative': 'Safe to use',
    'Trace': 'Monitor',
    'Weak_Positive': 'Check the cow',
    'Distinct_Positive': 'Veterinary care',
    'Definite_Positive': 'Reject the milk'
}

print("Prediction Examples:")
print("-" * 80)

for i, reading in enumerate(example_readings):
    input_data = np.array([[reading['ph'], reading['turbidity'], reading['ec'], 
                           reading['protein'], reading['scc']]])
    
    input_data_scaled = scaler.transform(input_data)
    
    prediction = best_model.predict(input_data_scaled)[0]
    quality_category = label_encoder.inverse_transform([prediction])[0]
    
    action = action_mapping.get(quality_category, 'Unknown')
    
    print(f"Example {i+1}:")
    print(f"Input: {reading}")
    print(f"Predicted Quality: {quality_category}")
    print(f"Recommended Action: {action}")
    print("-" * 80)"""))

with open('notebooks/milk_quality_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Fixed notebook created successfully at notebooks/milk_quality_analysis.ipynb")
