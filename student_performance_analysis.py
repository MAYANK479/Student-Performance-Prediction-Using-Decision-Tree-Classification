# Student Performance Analysis
# Subject: Data Mining and Data Warehousing
# Algorithm: Decision Tree Classification
# Dataset: UCI Student Performance Dataset (student-mat.csv)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

# output folder for saving graphs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(SCRIPT_DIR, "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

# ---- Step 1: Load Dataset ----
print("STEP 1: LOADING DATASET")

DATA_PATH = os.path.join(SCRIPT_DIR, "archive", "student-mat.csv")
df = pd.read_csv(DATA_PATH, sep=';')

print("\nFirst 5 rows:")
print(df.head())

print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nStatistical summary:\n{df.describe()}")


# ---- Step 2: Data Preprocessing ----
print("\nSTEP 2: DATA PREPROCESSING")

# check missing values
print("\nMissing values:")
missing = df.isnull().sum()
print(missing)
total_missing = missing.sum()
print(f"Total missing: {total_missing}")

if total_missing == 0:
    print("No missing values found.")
else:
    print("Handling missing values...")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    print("Done.")

# label encoding for categorical columns
print("\nEncoding categorical columns...")
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_columns}")

le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
    print(f"  Encoded: {col}")

print("Label encoding complete.")

# create target variable (pass/fail based on G3 >= 10)
print("\nCreating target variable 'Result'...")
df['Result'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
pass_count = df['Result'].sum()
fail_count = len(df) - pass_count
print(f"Pass (G3 >= 10): {pass_count} students ({pass_count/len(df)*100:.1f}%)")
print(f"Fail (G3 < 10):  {fail_count} students ({fail_count/len(df)*100:.1f}%)")

print(f"\nDataset after preprocessing:")
print(df.head())


# ---- Step 3: Exploratory Data Analysis ----
print("\nSTEP 3: EXPLORATORY DATA ANALYSIS")

# 3.1 Distribution of G3 (Final Grade)
print("\nPlotting G3 distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['G3'], bins=20, kde=True, color='steelblue', edgecolor='black', ax=ax)
ax.set_title('Distribution of Final Grade (G3)', fontsize=16, fontweight='bold')
ax.set_xlabel('Final Grade (G3)', fontsize=13)
ax.set_ylabel('Frequency', fontsize=13)
ax.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Pass/Fail Threshold (G3=10)')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'g3_distribution.png'))
plt.close()
print("Saved: graphs/g3_distribution.png")

# 3.2 Pass/Fail count plot
print("\nPlotting pass/fail distribution...")
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#e74c3c', '#2ecc71']
result_counts = df['Result'].value_counts().sort_index()
bars = ax.bar(['Fail (0)', 'Pass (1)'], result_counts.values, color=colors, edgecolor='black', width=0.5)
for bar, count in zip(bars, result_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.set_title('Count of Pass vs Fail Students', fontsize=16, fontweight='bold')
ax.set_xlabel('Result', fontsize=13)
ax.set_ylabel('Number of Students', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'pass_fail_countplot.png'))
plt.close()
print("Saved: graphs/pass_fail_countplot.png")

# 3.3 Correlation heatmap
print("\nPlotting correlation heatmap...")
fig, ax = plt.subplots(figsize=(18, 14))
correlation = df.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 7},
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap of All Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'correlation_heatmap.png'))
plt.close()
print("Saved: graphs/correlation_heatmap.png")

# 3.4 Study time vs result
print("\nPlotting study time vs result...")
fig, ax = plt.subplots(figsize=(10, 6))
studytime_result = df.groupby('studytime')['Result'].mean() * 100
bars = ax.bar(studytime_result.index, studytime_result.values,
              color=['#3498db', '#2ecc71', '#e67e22', '#9b59b6'],
              edgecolor='black', width=0.5)
for bar, val in zip(bars, studytime_result.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('Pass Rate by Weekly Study Time', fontsize=16, fontweight='bold')
ax.set_xlabel('Weekly Study Time\n(1: <2hrs, 2: 2-5hrs, 3: 5-10hrs, 4: >10hrs)', fontsize=12)
ax.set_ylabel('Pass Rate (%)', fontsize=13)
ax.set_ylim(0, 105)
ax.set_xticks([1, 2, 3, 4])
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'studytime_vs_result.png'))
plt.close()
print("Saved: graphs/studytime_vs_result.png")

# 3.5 Failures vs result
print("\nPlotting failures vs result...")
fig, ax = plt.subplots(figsize=(10, 6))
failures_result = df.groupby('failures')['Result'].mean() * 100
bars = ax.bar(failures_result.index, failures_result.values,
              color=['#2ecc71', '#f39c12', '#e74c3c', '#c0392b'],
              edgecolor='black', width=0.5)
for bar, val in zip(bars, failures_result.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('Pass Rate by Number of Past Failures', fontsize=16, fontweight='bold')
ax.set_xlabel('Number of Past Class Failures', fontsize=13)
ax.set_ylabel('Pass Rate (%)', fontsize=13)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'failures_vs_result.png'))
plt.close()
print("Saved: graphs/failures_vs_result.png")


# ---- Step 4: Feature Selection ----
print("\nSTEP 4: FEATURE SELECTION")

# drop G3 since Result is derived from it
X = df.drop(columns=['G3', 'Result'])
y = df['Result']

print(f"Features: {X.shape[1]} columns")
print(f"Feature names: {list(X.columns)}")
print(f"Target: Result (1=Pass, 0=Fail)")
print(f"Total samples: {len(y)}")


# ---- Step 5: Train-Test Split ----
print("\nSTEP 5: TRAIN-TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Testing set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")


# ---- Step 6: Decision Tree Classification ----
print("\nSTEP 6: DECISION TREE CLASSIFICATION")

dt_classifier = DecisionTreeClassifier(
    criterion='gini',
    random_state=42,
    max_depth=5
)

dt_classifier.fit(X_train, y_train)
print("Model trained successfully.")
print(f"Criterion: Gini Index")
print(f"Max Depth: {dt_classifier.get_depth()}")
print(f"Number of Leaves: {dt_classifier.get_n_leaves()}")

y_pred = dt_classifier.predict(X_test)
print("Predictions done on test set.")


# ---- Step 7: Model Evaluation ----
print("\nSTEP 7: MODEL EVALUATION")

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"  True Negatives  (correctly predicted Fail): {cm[0][0]}")
print(f"  False Positives (Fail predicted as Pass):   {cm[0][1]}")
print(f"  False Negatives (Pass predicted as Fail):   {cm[1][0]}")
print(f"  True Positives  (correctly predicted Pass): {cm[1][1]}")

# classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=['Fail (0)', 'Pass (1)'])
print(report)

# confusion matrix heatmap
print("Plotting confusion matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail (0)', 'Pass (1)'],
            yticklabels=['Fail (0)', 'Pass (1)'], annot_kws={'size': 20},
            linewidths=2, linecolor='black', ax=ax)
ax.set_title(f'Confusion Matrix (Accuracy: {accuracy*100:.2f}%)', fontsize=16, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('Actual Label', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'confusion_matrix.png'))
plt.close()
print("Saved: graphs/confusion_matrix.png")


# ---- Step 8: Feature Importance ----
print("\nSTEP 8: FEATURE IMPORTANCE")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_classifier.feature_importances_
}).sort_values(by='Importance', ascending=True)

print("\nTop 10 important features:")
top_features = feature_importance.tail(10)
for _, row in top_features.iterrows():
    bar = '█' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:>12s}: {row['Importance']:.4f} {bar}")

# plot feature importance
print("\nPlotting feature importance...")
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(feature_importance)))
ax.barh(feature_importance['Feature'], feature_importance['Importance'],
        color=colors, edgecolor='black', height=0.7)
ax.set_title('Feature Importance in Decision Tree Model', fontsize=16, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=13)
ax.set_ylabel('Features', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'feature_importance.png'))
plt.close()
print("Saved: graphs/feature_importance.png")


# ---- Step 9: Decision Tree Visualization ----
print("\nSTEP 9: DECISION TREE VISUALIZATION")

print("Plotting decision tree...")
fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
plot_tree(
    dt_classifier,
    feature_names=list(X.columns),
    class_names=['Fail', 'Pass'],
    filled=True,
    rounded=True,
    fontsize=7,
    proportion=True,
    ax=ax
)
ax.set_title('Decision Tree Visualization', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'decision_tree.png'), dpi=100, bbox_inches='tight')
plt.close()
print("Saved: graphs/decision_tree.png")


# ---- Summary ----
print("\n" + "=" * 50)
print("PROJECT COMPLETE")
print("=" * 50)
print(f"""
Dataset:        student-mat.csv ({df.shape[0]} students)
Features Used:  {X.shape[1]}
Algorithm:      Decision Tree (Gini, max_depth=5)
Train/Test:     70/30 split
Accuracy:       {accuracy * 100:.2f}%
Tree Depth:     {dt_classifier.get_depth()}
Tree Leaves:    {dt_classifier.get_n_leaves()}

Graphs saved in: {GRAPH_DIR}/
  1. g3_distribution.png
  2. pass_fail_countplot.png
  3. correlation_heatmap.png
  4. studytime_vs_result.png
  5. failures_vs_result.png
  6. confusion_matrix.png
  7. feature_importance.png
  8. decision_tree.png
""")
