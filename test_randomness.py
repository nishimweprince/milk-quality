import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
n_samples = 10000

ph_values = np.random.uniform(5.5, 7.5, n_samples)
turbidity = np.random.uniform(0.5, 30.0, n_samples)
ec_values = np.random.uniform(3.5, 7.0, n_samples)
protein_content = np.random.uniform(1.5, 4.0, n_samples)

scc_distribution = np.random.choice(['low', 'medium', 'high', 'very_high', 'extreme'], 
                                  size=n_samples, 
                                  p=[0.4, 0.3, 0.15, 0.1, 0.05])

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

quality_categories = []
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

df = pd.DataFrame({
    'pH': ph_values,
    'Turbidity': turbidity,
    'EC': ec_values,
    'Protein': protein_content,
    'SCC': scc_values,
    'MilkQuality': quality_categories
})

print('Quality Categories Distribution:')
print(df['MilkQuality'].value_counts())
print(df['MilkQuality'].value_counts() / len(df) * 100)

X = df[['pH', 'Turbidity', 'EC', 'Protein', 'SCC']]
y = df['MilkQuality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'\nRandom Forest Accuracy: {accuracy:.4f}')
print('This is a more realistic accuracy with added noise.')
