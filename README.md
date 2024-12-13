# Credit Card Fraud Detection Project

## Overview

This project involves detecting fraudulent transactions in a credit card dataset using machine learning techniques. The dataset, available on Kaggle, contains transactions labeled as fraudulent (1) or non-fraudulent (0). The objective is to build a classification model that identifies fraudulent transactions with high accuracy.

## Project Structure

project_root/

├── src/ 

│ ├── config.py             # Main configuration file containing dataset URLs and essential constants.  

│ ├── data_preprocessing.py # Core script for data processing, analysis, and model training.  

├── notebooks/

│ └── exploratory_analysis.ipynb # Exploratory Data Analysis (EDA) and experiments

├── images/ # Folder containing generated graphs

│ ├── fraud_vs_nonfraud.png

│ ├── roc_curve.png

│ ├── confusion_matrix.png

├── venv/ # Virtual environment (dependencies)

├── README.md # Project documentation

├── requirements.txt # Project dependencies

## Dataset

- **Source**: Kaggle - Credit Card Fraud Detection Dataset
- **Description**: Contains 284,807 transactions, where 0.17% are fraudulent.
- **Features**:
  - Numerical features V1, V2, ... V28 (resulting from PCA transformation)
  - Time: Seconds elapsed between this and the first transaction
  - Amount: Transaction amount
  - Class: Target variable (0 = Non-Fraud, 1 = Fraud)

## Key Steps

### Data Preprocessing:

- Handle missing and duplicate values.
- Normalize/scale transaction amounts.

### Exploratory Data Analysis (EDA):

- Analyze the distribution of fraudulent vs. non-fraudulent transactions.
- Visualize transaction amounts.

### Model Training:

- Train a Random Forest Classifier on the processed data.
- Evaluate performance using metrics like accuracy, confusion matrix, and ROC curve.

### Model Saving:

- Save the trained model for future predictions using joblib.

## Results

### Distribution of Transactions

### ROC Curve

The ROC curve shows the trade-off between the true positive rate and false positive rate. AUC = 0.99 indicates excellent performance.

### Confusion Matrix

The confusion matrix highlights the model’s ability to distinguish fraudulent transactions.

## Technologies Used

### Python Libraries:

- pandas, numpy for data manipulation and analysis
- matplotlib, seaborn for data visualization
- scikit-learn for machine learning and evaluation

### Tools:

- Jupyter Notebook for interactive data exploration
- Kaggle API for dataset access

## How to Run

1. Clone this repository.
2. Install dependencies using:
   "pip install -r requirements.txt"
3. Download the dataset from Kaggle and place it in the appropriate directory, or use kagglehub to automate the download.
4. Run the data_preprocessing.py script for model training:
  "python src/data_preprocessing.py"
5. View results in the images/ folder.

## Future Work

- Experiment with other classifiers like XGBoost or Neural Networks.
- Optimize hyperparameters for better accuracy.
- Explore anomaly detection methods for unsupervised learning.

## Acknowledgments

Special thanks to Kaggle for providing the dataset.
