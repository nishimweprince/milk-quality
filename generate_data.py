from ml_processor import MilkQualityML

def main():
    # Create an instance of MilkQualityML
    ml = MilkQualityML()
    
    # Generate and save the data
    print("Generating training data...")
    df = ml.generate_training_data(n_samples=200000)
    
    # Print some statistics about the generated data
    print("\nData Generation Complete!")
    print("\nQuality Categories Distribution:")
    print(df['MilkQuality'].value_counts())
    print("\nAction Distribution:")
    print(df['Action'].value_counts())
    print("\nData has been saved to: data/milk_quality_training_data.csv")

if __name__ == "__main__":
    main() 