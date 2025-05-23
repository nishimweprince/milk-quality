#!/usr/bin/env python

# 



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





def generate_training_data(n_samples=200000):
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
    
    return pd.DataFrame({
        'pH': ph_values,
        'Turbidity': turbidity,
        'EC': ec_values,
        'Protein': protein_content,
        'SCC': scc_values,
        'MilkQuality': quality_categories,
        'Action': actions
    })

df = generate_training_data()
df.head()





quality_counts = df['MilkQuality'].value_counts()
print("Milk Quality Distribution:")
print(quality_counts)
print("\nPercentage Distribution:")
print(quality_counts / len(df) * 100)

plt.figure(figsize=(12,6))
sns.countplot(data=df, x='MilkQuality', order=df['MilkQuality'].value_counts().index, palette='viridis')
plt.title('Milk Quality Distribution')
plt.ylabel('Count')
plt.xlabel('Quality Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





df_sample = df.sample(1000, random_state=42)

sns.pairplot(df_sample, hue='MilkQuality', vars=['pH', 'Turbidity', 'EC', 'Protein', 'SCC'])
plt.suptitle('Pairplot of Sampled Milk Data', y=1.02)
plt.show()




features = ['pH', 'Turbidity', 'EC', 'Protein', 'SCC']

fig, axes = plt.subplots(len(features), 1, figsize=(12, 15))
for i, feature in enumerate(features):
    sns.boxplot(x='MilkQuality', y=feature, data=df_sample, ax=axes[i])
    axes[i].set_title(f'{feature} by Milk Quality')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()





X = df[['pH', 'Turbidity', 'EC', 'Protein', 'SCC']]
y = df['MilkQuality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Label Encoding Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {i}")





X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")





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
    print(f"\nTraining {name}...")
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

print(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")





y_pred_best = best_model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', 
            xticklabels=list(label_encoder.classes_), 
            yticklabels=list(label_encoder.classes_), 
            cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()





if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    features = ['pH', 'Turbidity', 'EC', 'Protein', 'SCC']
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    print("Feature Importance:")
    for index, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
else:
    print(f"The {best_model_name} model does not provide feature importance information.")





plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = list(results.values())

bars = plt.bar(model_names, accuracies, color='skyblue')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.9, 1.0)  # Adjust as needed based on your results

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()





from sklearn.model_selection import cross_val_score

cv_results = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
    cv_results[name] = cv_scores
    print(f"{name} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

plt.figure(figsize=(10, 6))
plt.boxplot([cv_results[name] for name in model_names], labels=model_names)
plt.title('Cross-Validation Results')
plt.ylabel('Accuracy')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()





os.makedirs('models', exist_ok=True)

joblib.dump(best_model, 'models/milk_quality_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')

print(f"Best model ({best_model_name}) and preprocessing components saved successfully.")





example_readings = [
    {'ph': 6.8, 'turbidity': 2.5, 'ec': 4.5, 'protein': 3.2, 'scc': 150000},
    {'ph': 6.3, 'turbidity': 6.0, 'ec': 5.2, 'protein': 2.9, 'scc': 350000},
    {'ph': 5.8, 'turbidity': 15.0, 'ec': 6.5, 'protein': 2.2, 'scc': 3000000}
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
    print("-" * 80)
