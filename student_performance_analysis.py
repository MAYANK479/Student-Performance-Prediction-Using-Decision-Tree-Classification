import os
from itertools import combinations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MPLCONFIGDIR = os.path.join(SCRIPT_DIR, ".matplotlib")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)

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
import warnings
warnings.filterwarnings('ignore')

# using Agg backend so plots can be saved directly without opening a window
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

# save all plots in one folder so they are easy to use in the report
GRAPH_DIR = os.path.join(SCRIPT_DIR, "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "analysis_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def studytime_label(value):
    mapping = {
        1: "<2 hrs",
        2: "2-5 hrs",
        3: "5-10 hrs",
        4: ">10 hrs"
    }
    return mapping.get(value, str(value))


def failures_label(value):
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    return "2+"


def absences_bucket(value):
    if value <= 2:
        return "Low"
    if value <= 10:
        return "Medium"
    return "High"


def age_bucket(value):
    if value <= 16:
        return "<=16"
    if value <= 18:
        return "17-18"
    return "19+"


def grade_bucket(value):
    if value < 10:
        return "Low"
    if value < 15:
        return "Medium"
    return "High"


def build_transactions(dataframe):
    transactions = []

    for _, row in dataframe.iterrows():
        transactions.append({
            f"Sex={row['sex']}",
            f"School={row['school']}",
            f"Address={row['address']}",
            f"FamilySize={row['famsize']}",
            f"SchoolSupport={row['schoolsup']}",
            f"FamilySupport={row['famsup']}",
            f"HigherEducation={row['higher']}",
            f"Internet={row['internet']}",
            f"Activities={row['activities']}",
            f"Romantic={row['romantic']}",
            f"StudyTime={studytime_label(row['studytime'])}",
            f"Failures={failures_label(row['failures'])}",
            f"Absences={absences_bucket(row['absences'])}",
            f"AgeGroup={age_bucket(row['age'])}",
            f"G1Level={grade_bucket(row['G1'])}",
            f"G2Level={grade_bucket(row['G2'])}",
            f"Result={'Pass' if row['Result'] == 1 else 'Fail'}"
        })

    return transactions


def apriori(transactions, min_support=0.18, max_length=3):
    total_transactions = len(transactions)
    frequent_itemsets = {}

    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            itemset = frozenset([item])
            item_counts[itemset] = item_counts.get(itemset, 0) + 1

    current_level = {
        itemset: count / total_transactions
        for itemset, count in item_counts.items()
        if (count / total_transactions) >= min_support
    }
    frequent_itemsets.update(current_level)

    itemset_size = 2
    while current_level and itemset_size <= max_length:
        current_keys = list(current_level.keys())
        candidate_itemsets = set()

        for i in range(len(current_keys)):
            for j in range(i + 1, len(current_keys)):
                union_set = current_keys[i] | current_keys[j]
                if len(union_set) != itemset_size:
                    continue

                subsets = [frozenset(subset) for subset in combinations(union_set, itemset_size - 1)]
                if all(subset in current_level for subset in subsets):
                    candidate_itemsets.add(union_set)

        candidate_counts = {itemset: 0 for itemset in candidate_itemsets}
        for transaction in transactions:
            for itemset in candidate_itemsets:
                if itemset.issubset(transaction):
                    candidate_counts[itemset] += 1

        current_level = {
            itemset: count / total_transactions
            for itemset, count in candidate_counts.items()
            if (count / total_transactions) >= min_support
        }
        frequent_itemsets.update(current_level)
        itemset_size += 1

    return frequent_itemsets


def generate_association_rules(frequent_itemsets, min_confidence=0.65):
    rules = []

    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        for antecedent_size in range(1, len(itemset)):
            for antecedent in combinations(itemset, antecedent_size):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent

                antecedent_support = frequent_itemsets.get(antecedent)
                consequent_support = frequent_itemsets.get(consequent)
                if not antecedent_support or not consequent_support:
                    continue

                confidence = support / antecedent_support
                lift = confidence / consequent_support

                if confidence >= min_confidence:
                    rules.append({
                        'Antecedent': ', '.join(sorted(antecedent)),
                        'Consequent': ', '.join(sorted(consequent)),
                        'Support': support,
                        'Confidence': confidence,
                        'Lift': lift
                    })

    return pd.DataFrame(rules).sort_values(
        by=['Lift', 'Confidence', 'Support'],
        ascending=False
    ).reset_index(drop=True) if rules else pd.DataFrame()

print("STEP 1: LOADING DATASET")

DATA_PATH = os.path.join(SCRIPT_DIR, "archive", "student-mat.csv")
# the dataset uses semicolon as separator instead of comma
df = pd.read_csv(DATA_PATH, sep=';')

print("\nFirst 5 rows:")
print(df.head())

print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nStatistical summary:\n{df.describe()}")


print("\nSTEP 2: DATA PREPROCESSING")

print("\nMissing values:")
missing = df.isnull().sum()
print(missing)
total_missing = missing.sum()
print(f"Total missing: {total_missing}")

# if any values are missing, fill text columns with mode and numeric columns with median
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

raw_df = df.copy()
raw_df['Result'] = raw_df['G3'].apply(lambda x: 1 if x >= 10 else 0)

print("\nEncoding categorical columns...")
df = raw_df.copy()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_columns}")

# machine learning models need numeric input, so text values are encoded first
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
    print(f"  Encoded: {col}")

print("Label encoding complete.")

print("\nCreating target variable 'Result'...")
# final grade 10 or above is considered pass
df['Result'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
pass_count = df['Result'].sum()
fail_count = len(df) - pass_count
print(f"Pass (G3 >= 10): {pass_count} students ({pass_count/len(df)*100:.1f}%)")
print(f"Fail (G3 < 10):  {fail_count} students ({fail_count/len(df)*100:.1f}%)")

print(f"\nDataset after preprocessing:")
print(df.head())


print("\nSTEP 3: EXPLORATORY DATA ANALYSIS")

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

print("\nPlotting pass/fail distribution...")
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#e74c3c', '#2ecc71']
# value_counts gives total students in each class
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

print("\nPlotting study time vs result...")
fig, ax = plt.subplots(figsize=(10, 6))
# mean of Result gives pass rate because pass is stored as 1 and fail as 0
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

print("\nPlotting failures vs result...")
fig, ax = plt.subplots(figsize=(10, 6))
# this shows how previous failures affect current pass percentage
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


print("\nSTEP 4: OLAP ANALYSIS")

print("\nOLAP Query 1: Pass rate by gender and study time")
olap_pass_rate = raw_df.pivot_table(
    values='Result',
    index='sex',
    columns='studytime',
    aggfunc='mean'
) * 100
olap_pass_rate.index = [f"Sex={value}" for value in olap_pass_rate.index]
olap_pass_rate.columns = [studytime_label(value) for value in olap_pass_rate.columns]
olap_pass_rate = olap_pass_rate.round(2)
print(olap_pass_rate)
olap_pass_rate.to_csv(os.path.join(OUTPUT_DIR, 'olap_pass_rate_by_sex_studytime.csv'))
print("Saved: analysis_outputs/olap_pass_rate_by_sex_studytime.csv")

print("\nOLAP Query 2: Average grades by school and gender")
olap_grade_summary = raw_df.pivot_table(
    values=['G1', 'G2', 'G3'],
    index='school',
    columns='sex',
    aggfunc='mean'
).round(2)
print(olap_grade_summary)
olap_grade_summary.to_csv(os.path.join(OUTPUT_DIR, 'olap_grade_summary_school_sex.csv'))
print("Saved: analysis_outputs/olap_grade_summary_school_sex.csv")

print("\nOLAP Query 3: Pass rate and student count by internet and higher education goal")
olap_support = raw_df.groupby(['internet', 'higher']).agg(
    Student_Count=('Result', 'size'),
    Pass_Rate=('Result', lambda values: round(values.mean() * 100, 2)),
    Avg_G3=('G3', 'mean')
).round(2)
print(olap_support)
olap_support.to_csv(os.path.join(OUTPUT_DIR, 'olap_support_factors.csv'))
print("Saved: analysis_outputs/olap_support_factors.csv")

print("\nOLAP Query 4: Multi-dimensional cube summary")
olap_cube = raw_df.groupby(['studytime', 'failures', 'internet']).agg(
    Student_Count=('Result', 'size'),
    Pass_Rate=('Result', lambda values: round(values.mean() * 100, 2)),
    Avg_G3=('G3', 'mean')
).reset_index().round(2)
olap_cube['studytime'] = olap_cube['studytime'].apply(studytime_label)
olap_cube['failures'] = olap_cube['failures'].apply(failures_label)
print(olap_cube.head(12))
olap_cube.to_csv(os.path.join(OUTPUT_DIR, 'olap_cube_studytime_failures_internet.csv'), index=False)
print("Saved: analysis_outputs/olap_cube_studytime_failures_internet.csv")


print("\nSTEP 5: APRIORI ASSOCIATION ANALYSIS")

transactions = build_transactions(raw_df)
frequent_itemsets = apriori(transactions, min_support=0.18, max_length=3)
frequent_itemsets_df = pd.DataFrame([
    {
        'Itemset': ', '.join(sorted(itemset)),
        'Length': len(itemset),
        'Support': round(support, 4)
    }
    for itemset, support in frequent_itemsets.items()
]).sort_values(by=['Length', 'Support'], ascending=[True, False]).reset_index(drop=True)

print("\nTop frequent itemsets:")
print(frequent_itemsets_df.head(15))
frequent_itemsets_df.to_csv(os.path.join(OUTPUT_DIR, 'apriori_frequent_itemsets.csv'), index=False)
print("Saved: analysis_outputs/apriori_frequent_itemsets.csv")

association_rules_df = generate_association_rules(frequent_itemsets, min_confidence=0.65)
result_rules_df = association_rules_df[
    association_rules_df['Consequent'].str.contains('Result=')
].head(15) if not association_rules_df.empty else pd.DataFrame()

if result_rules_df.empty:
    print("\nNo strong Result-based association rules found with current thresholds.")
else:
    display_rules = result_rules_df.copy()
    display_rules[['Support', 'Confidence', 'Lift']] = display_rules[['Support', 'Confidence', 'Lift']].round(4)
    print("\nTop association rules linked with Result:")
    print(display_rules)

association_rules_df.to_csv(os.path.join(OUTPUT_DIR, 'apriori_association_rules.csv'), index=False)
print("Saved: analysis_outputs/apriori_association_rules.csv")


print("\nSTEP 6: FEATURE SELECTION")

# remove G3 because Result is created from it
X = df.drop(columns=['G3', 'Result'])
y = df['Result']

print(f"Features: {X.shape[1]} columns")
print(f"Feature names: {list(X.columns)}")
print(f"Target: Result (1=Pass, 0=Fail)")
print(f"Total samples: {len(y)}")


print("\nSTEP 7: TRAIN-TEST SPLIT")

# 70% data is used for training and 30% for testing
split_seed = 39
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=split_seed, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Testing set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"Train-Test Split Seed: {split_seed}")


print("\nSTEP 8: DECISION TREE CLASSIFICATION")

# max_depth=5 keeps the tree simple enough to understand during presentation
dt_classifier = DecisionTreeClassifier(
    criterion='gini',
    random_state=42,
    max_depth=5,
    min_samples_split=4
)

dt_classifier.fit(X_train, y_train)
print("Model trained successfully.")
print(f"Criterion: Gini Index")
print(f"Max Depth: {dt_classifier.get_depth()}")
print(f"Min Samples Split: 4")
print(f"Number of Leaves: {dt_classifier.get_n_leaves()}")

# predictions are made only on the test data
y_pred = dt_classifier.predict(X_test)
print("Predictions done on test set.")


print("\nSTEP 9: MODEL EVALUATION")

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"  True Negatives  (correctly predicted Fail): {cm[0][0]}")
print(f"  False Positives (Fail predicted as Pass):   {cm[0][1]}")
print(f"  False Negatives (Pass predicted as Fail):   {cm[1][0]}")
print(f"  True Positives  (correctly predicted Pass): {cm[1][1]}")

print("\nClassification Report:")
# classification report gives precision, recall and f1-score for both classes
report = classification_report(y_test, y_pred, target_names=['Fail (0)', 'Pass (1)'])
print(report)

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


print("\nSTEP 10: FEATURE IMPORTANCE")

# sorting makes the bar chart easier to read from low to high
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_classifier.feature_importances_
}).sort_values(by='Importance', ascending=True)

print("\nTop 10 important features:")
top_features = feature_importance.tail(10)
for _, row in top_features.iterrows():
    bar = '█' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:>12s}: {row['Importance']:.4f} {bar}")

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


print("\nSTEP 11: DECISION TREE VISUALIZATION")

print("Plotting decision tree...")
# plot_tree helps show the decision rules followed by the model
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
Extra Mining:   OLAP summaries + Apriori association rules

Graphs saved in: {GRAPH_DIR}/
  1. g3_distribution.png
  2. pass_fail_countplot.png
  3. correlation_heatmap.png
  4. studytime_vs_result.png
  5. failures_vs_result.png
  6. confusion_matrix.png
  7. feature_importance.png
  8. decision_tree.png

Analysis tables saved in: {OUTPUT_DIR}/
  1. olap_pass_rate_by_sex_studytime.csv
  2. olap_grade_summary_school_sex.csv
  3. olap_support_factors.csv
  4. olap_cube_studytime_failures_internet.csv
  5. apriori_frequent_itemsets.csv
  6. apriori_association_rules.csv
""")
