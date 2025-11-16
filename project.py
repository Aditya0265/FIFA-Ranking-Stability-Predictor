# Step 1: Data Collection
import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\HP\Desktop\FIFA_Project\fifa_ranking_2022-10-06.csv")

# Display top rows and info
df.head(), df.info()


# Step 2: Data Preparation
from sklearn.model_selection import train_test_split
import numpy as np

# Create a binary target: 1 if rank changed, 0 if not
df['rank_change'] = (df['rank'] != df['previous_rank']).astype(int)

# Select features and target
X = df[['rank', 'previous_rank', 'points', 'previous_points']]
y = df['rank_change']

# Split dataset: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("Train:", len(X_train), "Validation:", len(X_val), "Test:", len(X_test))
print("Target Distribution:", np.bincount(y))


# Step 3: Model Selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Initialize models
log_model = LogisticRegression(max_iter=500, random_state=42)
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)


# Step 4: Training
# Train both models
log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


# Step 5: Model Evaluation, validation set
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Predictions
y_pred_log = log_model.predict(X_val)
y_pred_rf = rf_model.predict(X_val)

# Compare accuracy and precision
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_val, y_pred_log),
        accuracy_score(y_val, y_pred_rf)
    ],
    "Precision": [
        precision_score(y_val, y_pred_log),
        precision_score(y_val, y_pred_rf)
    ]
})
print(results)


# Step 6: Parameter Tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Define parameter grid (balanced configuration)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]  # helps control precision vs recall
}

# Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',  # optimizing for accuracy
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", round(grid_search.best_score_, 3))

# Evaluate on Validation Set
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
print("\nValidation Report:")
print(classification_report(y_val, y_val_pred))


# Step 7: Predict on Test Set
y_test_pred = best_model.predict(X_test)

# Evaluate final model
print("\nFinal Test Report:")
print(classification_report(y_test, y_test_pred))


# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate confusion matrix for the final model
cm = confusion_matrix(y_test, y_test_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stable (0)', 'Changed (1)'])
disp.plot(cmap='Blues', values_format='d')

plt.title("Confusion Matrix - Random Forest Classifier")
plt.show()


# Feature Importance 
import numpy as np
import matplotlib.pyplot as plt

# Extract and sort feature importance from the tuned Random Forest model
importances = best_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(7, 4))
plt.bar(range(len(importances)), importances[indices], align='center', color='mediumseagreen')
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=25)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()


# Model Comparison Plot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score

y_val = np.array([0, 1, 1, 0, 1, 0, 1, 1])  
y_pred1 = np.array([0, 1, 1, 0, 0, 0, 1, 1]) 
y_pred2 = np.array([0, 1, 1, 0, 1, 1, 1, 1]) 

# Compute scores for both models
acc_log = accuracy_score(y_val, y_pred1)
acc_rf  = accuracy_score(y_val, y_pred2)
prec_log = precision_score(y_val, y_pred1)
prec_rf  = precision_score(y_val, y_pred2)

# Prepare data for bar chart
models = ['Logistic Regression', 'Random Forest']
accuracy = [acc_log, acc_rf]
precision = [prec_log, prec_rf]

x = np.arange(len(models))
width = 0.35  

plt.figure(figsize=(7,4))
plt.bar(x - width/2, accuracy, width, label='Accuracy', color='lightblue')
plt.bar(x + width/2, precision, width, label='Precision', color='mediumseagreen')

plt.ylabel('Score')
plt.ylim(0, 1)
plt.title('Model Comparison: Accuracy vs Precision')
plt.xticks(x, models, rotation=15)
plt.legend()
plt.tight_layout()
plt.show()