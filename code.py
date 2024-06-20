from ctypes import sizeof
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Step 1: Get all file names and their labels
def get_file_names_with_labels(directory):
    file_names = []
    labels = []
    
    # Loop through each file in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_names.append(file_name)
            label = file_name.split('_')[0]  # Extract the label before the first underscore
            labels.append(label)
    
    return file_names, labels

# Example usage:
csv_directory = 'C:/Users/acer/Documents/CSES/Data/'  # Adjust the directory path as needed  # Adjust the image directory path as needed

file_names, labels = get_file_names_with_labels(csv_directory)

# Print file names and their corresponding labels
for file_name, label in zip(file_names, labels):
    print("File:", file_name, "Label:", label)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(file_names, labels, test_size=0.2, random_state=7)

# Load and process training data
train_data = []
train_label = []
for file_name, label in zip(X_train, y_train):
    file_path = os.path.join(csv_directory, file_name)
    df = pd.read_csv(file_path)
    features = df.iloc[:, :7].values
    train_data.extend(features)
    train_label.extend([label] * len(features))

# Define the range of neighbors to try
neighbors = list(range(1, 11))  # Trying neighbors from 1 to 10

# Perform grid search to find the best number of neighbors
param_grid = {'n_neighbors': neighbors}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(train_data, train_label)

# Get the best number of neighbors
best_neighbors = grid_search.best_params_['n_neighbors']

# Train the model with the best number of neighbors
knn = KNeighborsClassifier(n_neighbors=best_neighbors)
knn.fit(train_data, train_label)

# Store the trained model for later use
models = {"KNeighborsClassifier": knn}

# Load and process test data, make predictions, and calculate accuracy
y_pred = []
test_data = []
test_label = []
incorrectly_predicted_files = []  # List to store file names that were incorrectly predicted
for file_name, label in zip(X_test, y_test):
    file_path = os.path.join(csv_directory, file_name)
    df = pd.read_csv(file_path)
    X_test_data = df.iloc[:, :7].values
    prediction = knn.predict(X_test_data)
    y_pred.extend(prediction)
    test_data.extend(X_test_data)
    test_label.extend([label] * len(X_test_data))
    
    # Check if prediction is correct
    incorrect_predictions = np.array(prediction) != label
    if np.any(incorrect_predictions):
        incorrectly_predicted_files.append((file_name, label))

# Calculate accuracy
print(incorrectly_predicted_files)
accuracy = accuracy_score(test_label, y_pred)
print("K-Nearest Neighbors Classifier Accuracy:", accuracy)
print("Best number of neighbors:", best_neighbors)

# Print out the incorrectly predicted instances

for file_name, true_label in incorrectly_predicted_files:
    print(file_name)
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate confusion matrix
conf_matrix = confusion_matrix(test_label, y_pred)

# Visualize confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
import pandas as pd
import cv2

def draw_connecting_lines(image_path, csv_path, img, offset_x, offset_y):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert center coordinates to integer
    df['center_x'] = df['center_x'].astype(int)
    df['center_y'] = df['center_y'].astype(int)
    
    # Plot connecting lines and mark points
    for i in range(len(df) - 1):
        point1 = (df.iloc[i]['center_x'] + offset_x, df.iloc[i]['center_y'] + offset_y)
        point2 = (df.iloc[i+1]['center_x'] + offset_x, df.iloc[i+1]['center_y'] + offset_y)
        cv2.line(img, point1, point2, (0, 0, 255), 2)
    
    # Mark the first point in blue
    first_point = (df.iloc[0]['center_x'] + offset_x, df.iloc[0]['center_y'] + offset_y)
    cv2.circle(img, first_point, 5, (255, 0, 0), -1)
    
    # Mark the last point in yellow
    last_point = (df.iloc[-1]['center_x'] + offset_x, df.iloc[-1]['center_y'] + offset_y)
    cv2.circle(img, last_point, 5, (0, 255, 255), -1)

# Example usage:
path = 'Data/'

# Read the base image
base_image = cv2.imread('sample12.jpg')

# Set initial offsets for placing points from different files
offset_x = 0
offset_y = 0
print(len(incorrectly_predicted_files))
for filename, labels in incorrectly_predicted_files:
    if filename.startswith('abnormal_12'):
        draw_connecting_lines(path + filename.replace('.csv', '.jpg'), path + filename, base_image, offset_x, offset_y)
        # Adjust offsets for the next file
        offset_x += 50  # Adjust as needed
        offset_y += 50  # Adjust as needed

# Save the final annotated image
cv2.imwrite('combined_abnormal_12_image.jpg', base_image)