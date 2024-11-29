
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv("heart_disease_data.csv")


print(data.head())

X = data.drop("target", axis=1) 
y = data["target"]             


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))


new_data = np.array([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 0, 0, 1]])  
new_data_scaled = scaler.transform(new_data)  
prediction = model.predict(new_data_scaled)
print("Heart Disease Prediction (1: Disease, 0: No Disease):", prediction[0])
