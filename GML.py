from train import model
import torch
from torch_geometric.data import Data
from dataset import AMLtoGraph
from sklearn.model_selection import train_test_split
import os
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Define the dataset path
dataset_path = r"C:\Users\HP\Desktop\AntiMoney\data"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = AMLtoGraph(dataset_path)
data = dataset[0]

# Ensure the model is in evaluation mode
model.load_state_dict(torch.load('best_model.pth'))

# Generate embeddings
with torch.no_grad():
    _, all_embeddings = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))

# Convert embeddings to numpy
embeddings = all_embeddings.cpu().numpy()

# Use embeddings as features for ML models
X = embeddings
Y = data.y.numpy()

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# File to save ML models results
results_dir = 'results'
gml_results_file = os.path.join(results_dir, 'gml_results.csv')

# Initialize the CSV file for ML models
with open(gml_results_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model', 'Accuracy', 'Confusion Matrix', 'Precision', 'Recall', 'F1-Score', 'Support'])

models = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    SVC(probability=True),
    LogisticRegression(),
    LinearDiscriminantAnalysis()
]

for clf in models:
    clf.fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Extract precision, recall, f1-score, and support for each class
    precision = report_dict['weighted avg']['precision']
    recall = report_dict['weighted avg']['recall']
    f1_score = report_dict['weighted avg']['f1-score']
    support = report_dict['weighted avg']['support']

    print(f" {clf.__class__.__name__} evaluation: ".center(100, "#"))
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print()

    # Save results to CSV file
    with open(gml_results_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            clf.__class__.__name__,
            accuracy,
            cm.tolist(),
            precision,
            recall,
            f1_score,
            support
        ])