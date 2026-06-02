import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("data/student_data.csv")

print("Dataset Preview")
print(data.head())

# Features
X = data[['study_hours', 'attendance_percentage', 'previous_score']]

# Target
y = data['pass_or_fail']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Training
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

# Sample Prediction
sample = pd.DataFrame(
    [[10, 80, 70]],
    columns=['study_hours', 'attendance_percentage', 'previous_score']
)

result = model.predict(sample)

print("\nPredicted Result:", result[0])