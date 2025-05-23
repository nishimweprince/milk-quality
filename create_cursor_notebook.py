import nbformat as nbf
import json
import re

# Create a new notebook
nb = nbf.v4.new_notebook()

# Read the Python script
with open('notebooks/milk_quality_analysis_updated.py', 'r') as f:
    script = f.read()

# Split the script into sections based on multiple blank lines
sections = re.split(r'\n{3,}', script)

# Add a title cell
nb['cells'].append(nbf.v4.new_markdown_cell('# Milk Quality Analysis and Machine Learning Pipeline\n\nThis notebook demonstrates the process of generating synthetic sensor data for milk quality analysis, performing exploratory data analysis (EDA), preprocessing, training multiple machine learning models, and evaluating their performance.'))

# Process each section
for i, section in enumerate(sections):
    if i == 0:  # Import section
        nb['cells'].append(nbf.v4.new_code_cell(section.strip()))
        continue
        
    # Check if section contains code
    if section.strip():
        # Add a markdown cell with a title based on content
        title = ""
        if "generate_training_data" in section:
            title = "## 1. Data Generation\n\nGenerate synthetic milk quality data with various parameters."
        elif "quality_counts" in section:
            title = "## 2. Data Distribution Analysis\n\nExamine the distribution of milk quality categories."
        elif "df_sample = df.sample" in section:
            title = "## 3. Exploratory Data Analysis\n\nVisualize relationships between features and milk quality."
        elif "features = [" in section:
            title = "## 4. Feature Analysis by Quality Category\n\nAnalyze how features vary across different milk quality categories."
        elif "X = df[" in section:
            title = "## 5. Data Preprocessing\n\nPrepare data for machine learning by scaling features and encoding labels."
        elif "X_train, X_test" in section:
            title = "## 6. Train/Test Split\n\nSplit data into training (80%) and testing (20%) sets."
        elif "models = {" in section:
            title = "## 7. Model Training and Evaluation\n\nTrain multiple machine learning models and compare their performance."
        elif "y_pred_best = best_model.predict" in section:
            title = "## 8. Confusion Matrix\n\nVisualize the confusion matrix for the best performing model."
        elif "if hasattr(best_model, 'feature_importances_')" in section:
            title = "## 9. Feature Importance Analysis\n\nAnalyze which features contribute most to the milk quality prediction."
        elif "plt.figure(figsize=(10, 6))\nmodel_names" in section:
            title = "## 10. Model Comparison\n\nCompare the accuracy of different machine learning models."
        elif "from sklearn.model_selection import cross_val_score" in section:
            title = "## 11. Cross-Validation\n\nPerform cross-validation to ensure model robustness."
        elif "os.makedirs('models'" in section:
            title = "## 12. Model Saving\n\nSave the best model and preprocessing components for deployment."
        elif "example_readings" in section:
            title = "## 13. Prediction Examples\n\nTest the model with example milk quality readings."
        
        if title:
            nb['cells'].append(nbf.v4.new_markdown_cell(title))
            
        # Add the code cell
        nb['cells'].append(nbf.v4.new_code_cell(section.strip()))

# Save the notebook
with open('notebooks/milk_quality_analysis.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print('Cursor-compatible notebook created successfully')
