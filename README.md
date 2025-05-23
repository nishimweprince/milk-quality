# Milk Quality Monitoring System

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/nishimweprince/milk-quality)

An IoT platform for online milk quality analysis and monitoring using Arduino sensors and machine learning.

## Overview

This project implements a system to measure different components in milk to evaluate its safety. It uses Arduino-based sensors to collect data on pH, turbidity, electrical conductivity (EC), protein content, and somatic cell count (SCC). The system then processes this data using machine learning models to classify milk quality and recommend appropriate actions.

## Features

- **Real-time Monitoring**: Collects and processes sensor data in real-time
- **Machine Learning Classification**: Uses multiple models to classify milk quality into categories
- **Web Interface**: Provides a web dashboard for visualizing milk quality data
- **Database Storage**: Stores historical milk quality data for analysis
- **Arduino Integration**: Supports Arduino connections on both Windows and Mac systems
- **Demo Mode**: Includes a demo mode for testing without physical sensors

## Milk Quality Categories

The system classifies milk into the following categories:

- **Negative**: Safe to use (SCC <= 200,000)
- **Trace**: Monitor (SCC 200,001-400,000)
- **Weak Positive**: Check the cow (SCC 400,001-1,200,000)
- **Distinct Positive**: Veterinary care needed (SCC 1,200,001-5,000,000)
- **Definite Positive**: Reject the milk (SCC > 5,000,000)

## Setup Instructions

### Prerequisites

- Python 3.6 or higher
- Arduino board with appropriate sensors
- MySQL database

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/milk-quality.git
   cd milk-quality
   ```

2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up the MySQL database:
   ```
   mysql -u root -p < milkquality.sql
   ```

4. Upload the Arduino code to your board (see Arduino folder for details)

### Configuration

- Update the database connection details in `insert_data.py` if needed
- Configure Arduino port settings if your device is not automatically detected

## Usage

1. Start the web server:
   ```
   python insert_data.py
   ```

2. Access the web interface at http://localhost:5001

3. To train new models or analyze data, run the Jupyter notebook:
   ```
   jupyter notebook notebooks/milk_quality_analysis.ipynb
   ```

## Machine Learning Pipeline

The system uses multiple machine learning models to classify milk quality:

1. **Data Collection**: Sensor data is collected from Arduino or generated in demo mode
2. **Preprocessing**: Data is scaled and normalized
3. **Model Training**: Multiple models (Random Forest, SVM, Logistic Regression, Gradient Boosting) are trained
4. **Model Evaluation**: Models are evaluated using accuracy, precision, recall, and F1-score
5. **Model Selection**: The best performing model is selected for deployment
6. **Prediction**: New sensor readings are classified using the selected model

## Project Structure

- `insert_data.py`: Web server and Arduino data collection
- `ml_processor.py`: Machine learning model training and prediction
- `generate_data.py`: Generate synthetic training data
- `notebooks/`: Jupyter notebooks for data analysis
- `models/`: Saved machine learning models
- `data/`: Generated data and datasets
- `templates/`: Web interface templates
- `milkquality.sql`: Database schema

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
