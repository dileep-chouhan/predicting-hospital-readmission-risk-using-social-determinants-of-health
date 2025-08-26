import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Generate synthetic patient data
num_patients = 500
data = {
    'Age': np.random.randint(30, 80, size=num_patients),
    'Gender': np.random.choice(['Male', 'Female'], size=num_patients),
    'Income': np.random.randint(20000, 150000, size=num_patients),
    'ChronicDisease': np.random.choice(['Yes', 'No'], size=num_patients, p=[0.3, 0.7]),
    'Insurance': np.random.choice(['Yes', 'No'], size=num_patients, p=[0.8, 0.2]),
    'HospitalVisitsLastYear': np.random.randint(0, 5, size=num_patients),
    'Readmitted': np.random.choice([0, 1], size=num_patients, p=[0.8, 0.2]) # 0: No, 1: Yes
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preprocessing ---
# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'ChronicDisease', 'Insurance'], drop_first=True)
# --- 3. Data Analysis and Model Building ---
# Split data into features (X) and target (y)
X = df.drop('Readmitted', axis=1)
y = df['Readmitted']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a logistic regression model
model = LogisticRegression(max_iter=1000) #increased max_iter to ensure convergence
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# --- 4. Model Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Readmitted', 'Readmitted'],
            yticklabels=['Not Readmitted', 'Readmitted'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
# Save the plot to a file
output_filename = 'confusion_matrix.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")