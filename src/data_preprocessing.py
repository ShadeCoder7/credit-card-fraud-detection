# Dataset URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For advanced data visualization
import kagglehub  # For downloading datasets from Kaggle
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn.ensemble import RandomForestClassifier  # For creating a classification model with Random Forest
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc  # For evaluating the model's performance
import warnings  # For handling warnings
import joblib  # For saving the trained model

warnings.filterwarnings("ignore")  # Ignore warnings

# Organize the data into a dataframe

# Download the dataset from Kaggle
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)

# Load the CSV file into a dataframe
data = pd.read_csv(f"{path}/creditcard.csv")

# Display the first 10 rows
data.head(10)

# Check for missing values in each column
missing_values = data.isnull().sum()

# Display the missing values per column
print("Missing values per column:")
print(missing_values)

# Check for duplicate rows in the dataframe
duplicate_rows = data.duplicated().sum()

# Display the number of duplicate rows
print(f"Number of duplicate rows: {duplicate_rows}")

# What percentage of transactions in the dataset are fraudulent?
# Count the total number of transactions
total_transactions = len(data)

# Count the number of fraudulent transactions (where "Class" is 1)
fraudulent_transactions = data[data['Class'] == 1].shape[0]

# Calculate the percentage of fraudulent transactions
fraudulent_percentage = (fraudulent_transactions / total_transactions) * 100

# Display the percentage of fraudulent transactions
print(f"Percentage of fraudulent transactions: {fraudulent_percentage:.2f}%")

# What is the average amount of fraudulent transactions in the dataset?
# Filter the fraudulent transactions
fraudulent_data = data[data['Class'] == 1]

# Calculate the average amount of fraudulent transactions
average_fraudulent_amount = fraudulent_data['Amount'].mean()

# Display the average amount of fraudulent transactions
print(f"Average amount of fraudulent transactions: {average_fraudulent_amount:.2f}")

# Visualize the distribution of fraudulent vs non-fraudulent transactions (Bar plot)
# Count total transactions
total_transactions = len(data)

# Count fraudulent transactions (where "Class" is 1)
fraudulent_transactions = data[data['Class'] == 1].shape[0]

# Calculate percentage of fraudulent transactions
fraudulent_percentage = (fraudulent_transactions / total_transactions) * 100

# Display the percentage of fraudulent transactions
print(f"Percentage of fraudulent transactions: {fraudulent_percentage:.2f}%")

# Visualize the distribution of fraudulent vs non-fraudulent transactions (Bar plot)
transaction_counts = data['Class'].value_counts()

plt.figure(figsize=(8, 6))
transaction_counts.plot(kind='bar', color=['red', 'green'])
plt.title('Number of Fraudulent vs Non-Fraudulent Transactions')
plt.xlabel('Class (0: No Fraud, 1: Fraud)')
plt.ylabel('Number of Transactions')
plt.xticks([0, 1], ['No Fraud', 'Fraud'], rotation=0)
plt.show()

# What is the distribution of amounts for fraudulent transactions?
# Separate the fraudulent transaction data
fraudulent_data = data[data['Class'] == 1]
# Display the distribution of the amounts of fraudulent transactions
plt.figure(figsize=(10, 6))
plt.hist(fraudulent_data['Amount'], bins=50, color='red', alpha=0.7)
plt.title('Distribution of Fraudulent Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Separate the features (X) and the target variable (y)
X = data.drop('Class', axis=1)  # Drop the 'Class' column to use the rest as features
y = data['Class']  # Use the 'Class' column as the target variable

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the dimensions of the resulting datasets
print(f"Training set size (X_train, y_train): {X_train.shape}, {y_train.shape}")
print(f"Test set size (X_test, y_test): {X_test.shape}, {y_test.shape}")

# Create and train the Random Forest Classifier model
rf_classifier = RandomForestClassifier(random_state=42)  # Initialize the Random Forest model

# Train the model with the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Display the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy in percentage format
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraude", "Fraude"], yticklabels=["No Fraude", "Fraude"])
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# Compute the ROC curve
fpr, tpr, _ = roc_curve(y_test, rf_classifier.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Área bajo la curva (AUC) = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.title("Curva ROC")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.legend(loc="lower right")
plt.show()

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to display the feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Importancia de las Características")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.show()

# Save the trained model
joblib.dump(rf_classifier, 'random_forest_model.pkl')