#!/usr/bin/env python
# coding: utf-8

# # Milk Quality Analysis and Machine Learning Pipeline
# 
# This notebook outlines the process of generating synthetic sensor data for milk quality analysis, performing exploratory data analysis (EDA), preprocessing, training a Random Forest classifier, and saving the model components for deployment.

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os


# ## 1. Generate Synthetic Training Data
# The synthetic data simulates sensor readings from milk quality monitoring equipment.

# In[ ]:


def generate_training_data(n_samples=200000):
    np.random.seed(42)
    ph_values = np.random.uniform(6.0, 7.5, n_samples)
    turbidity = np.random.uniform(0.5, 10.0, n_samples)
    ec_values = np.random.uniform(4.0, 6.0, n_samples)
    protein_content = np.random.uniform(1.8, 3.8, n_samples)
    
    w1, w2, C = 1.2, 2.0, 1000
    scc_values = (w1 * ec_values) + (w2 * turbidity) + C
    scc_values += np.random.normal(0, 500, n_samples)
    scc_values = np.abs(scc_values)
    
    quality_categories = []
    for scc in scc_values:
        if scc <= 200000:
            quality_categories.append('Negative')
        elif scc <= 400000:
            quality_categories.append('Trace')
        elif scc <= 1200000:
            quality_categories.append('Weak_Positive')
        elif scc <= 5000000:
            quality_categories.append('Distinct_Positive')
        else:
            quality_categories.append('Definite_Positive')

    return pd.DataFrame({
        'pH': ph_values,
        'Turbidity': turbidity,
        'EC': ec_values,
        'Protein': protein_content,
        'SCC': scc_values,
        'MilkQuality': quality_categories
    })

df = generate_training_data()
df.head()


# ## 2. Exploratory Data Analysis (EDA)
# Let's visualize the distribution of features and the target label.

# In[ ]:


sns.pairplot(df.sample(1000), hue='MilkQuality')
plt.suptitle('Pairplot of Sampled Milk Data', y=1.02)
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(data=df, x='MilkQuality', order=df['MilkQuality'].value_counts().index, palette='viridis')
plt.title('Milk Quality Distribution')
plt.ylabel('Count')
plt.xlabel('Quality Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## 3. Preprocessing
# We will scale the features and encode the target labels.

# In[ ]:


X = df[['pH', 'Turbidity', 'EC', 'Protein', 'SCC']]
y = df['MilkQuality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# ## 4. Train/Test Split
# Split the data for training and evaluation.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


# ## 5. Model Training
# We use a Random Forest Classifier.

# In[ ]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# ## 6. Evaluation
# We'll look at classification performance and a confusion matrix.

# In[ ]:


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# In[ ]:


conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# ## 7. Save Trained Artifacts
# Store the model and preprocessing components for use in production.

# In[ ]:


os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/milk_quality_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
print("Models saved successfully.")

