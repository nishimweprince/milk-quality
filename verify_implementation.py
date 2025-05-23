import pandas as pd
import numpy as np
from ml_processor import MilkQualityML

ml = MilkQualityML()

print('Generating test data...')
df = ml.generate_training_data(n_samples=1000)

print('\nQuality Categories Distribution:')
print(df['MilkQuality'].value_counts())

print('\nTraining models...')
ml.train_models()

print('\nModel evaluation complete.')
