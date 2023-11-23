# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

# Load the Iris dataset
# If you have a CSV file, you can use: iris_data = pd.read_csv('iris.csv')
iris_data = datasets.load_iris()
X = iris_data.data[:, 2:]  # Using only petal length and width
y = iris_data.target

# Explore and pre-visualize the data
# Visualize the data using scatter plot or any other suitable method

# Declare the variables X and Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the decision tree using Gini index
clf_gini = DecisionTreeClassifier(criterion='gini')
clf_gini.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(10, 8))
tree.plot_tree(clf_gini, feature_names=["petal length", "petal width"], class_names=iris_data.target_names, filled=True)
plt.show()

# Evaluate the decision tree
y_pred = clf_gini.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Display classification report
class_report = classification_report(y_test, y_pred, target_names=iris_data.target_names)
print("Classification Report:")
print(class_report)

# Predict new entries
new_entries = np.array([[2, 0.5], [2.5, 0.75]])
predictions = clf_gini.predict(new_entries)
print("Predictions for new entries:")
for i, pred in enumerate(predictions):
    print(f"Entry {i+1}: {iris_data.target_names[pred]}")
