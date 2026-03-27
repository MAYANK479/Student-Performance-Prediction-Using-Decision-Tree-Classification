# Student Performance Prediction Using Decision Tree Classification

This project applies a Decision Tree classifier to the UCI Student Performance Dataset to predict whether a student will pass or fail based on demographic, social, and academic features.

## Project Overview

- Dataset: UCI Student Performance Dataset (`student-mat.csv`)
- Language: Python
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
- Model: Decision Tree Classifier (CART with Gini Index)
- Objective: Predict student pass/fail outcomes and identify important factors influencing performance

## Features

- Data loading and preprocessing
- Label encoding for categorical features
- Exploratory data analysis with visualizations
- Decision Tree model training and evaluation
- Confusion matrix and classification report
- Feature importance analysis
- Decision tree visualization

## Dataset

The dataset used in this project is the UCI Student Performance Dataset, which contains student demographic, family, lifestyle, and academic information from Portuguese secondary schools.

File used:

- `archive/student-mat.csv`

## Results

- Accuracy: `86.55%`
- Tree Depth: `5`
- Leaves: `16`

The model uses a binary target variable:

- `Pass (1)`: `G3 >= 10`
- `Fail (0)`: `G3 < 10`

## Generated Outputs

The script saves the following visualizations in the `graphs/` folder:

- `g3_distribution.png`
- `pass_fail_countplot.png`
- `correlation_heatmap.png`
- `studytime_vs_result.png`
- `failures_vs_result.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `decision_tree.png`

## Project Structure

```text
.
|-- academic_report.md
|-- Project_Report.md
|-- student_performance_analysis.py
|-- archive/
|   |-- student-mat.csv
|   |-- student-por.csv
|   `-- student.txt
`-- graphs/
```

## How to Run

Run the analysis script with:

```bash
python -u "student_performance_analysis.py"
```

If you are running from another directory, use the full path to the script.

## Repository Description

Data mining project using the UCI Student Performance Dataset to predict student pass/fail outcomes with a Decision Tree classifier, EDA, visualizations, and an academic report.

## Author

Mayank Pandey
