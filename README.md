# Student Performance Prediction Using Decision Tree Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python" alt="Python Badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Decision%20Tree-f7931e?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn Badge" />
  <img src="https://img.shields.io/badge/Dataset-UCI%20Student%20Performance-2ea44f?style=for-the-badge" alt="Dataset Badge" />
  <img src="https://img.shields.io/badge/Status-Completed-1f883d?style=for-the-badge" alt="Status Badge" />
</p>

<p align="center">
  A data mining project that uses the UCI Student Performance Dataset to predict student pass/fail outcomes with a Decision Tree classifier, supported by EDA, visualizations, and an academic report.
</p>

---

## Overview

This project analyzes student academic data from Portuguese secondary schools and builds a Decision Tree model to predict whether a student will pass or fail. The workflow includes data preprocessing, exploratory analysis, model training, evaluation, feature importance analysis, and tree visualization.

## Highlights

- Predicts `Pass` or `Fail` using student demographic, academic, and lifestyle attributes
- Uses a `Decision Tree Classifier (CART)` with `Gini Index`
- Includes exploratory visual analysis and model evaluation plots
- Identifies the most influential features affecting student performance
- Produces ready-to-use graphs for reports and presentations

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Libraries | pandas, numpy, matplotlib, seaborn, scikit-learn |
| Dataset | UCI Student Performance Dataset |
| Model | Decision Tree Classification |

## Project Results

| Metric | Value |
|---|---|
| Accuracy | `86.55%` |
| Tree Depth | `5` |
| Number of Leaves | `16` |
| Dataset Size | `395` student records |
| Input Features | `32` |
| Target | `Pass (1)` / `Fail (0)` |

## Dataset

The project uses the `student-mat.csv` file from the UCI Student Performance Dataset. It contains student information related to:

- Demographics
- Family background
- Study habits
- Social behavior
- Health and absences
- Academic grades

Target definition:

- `Pass (1)` if `G3 >= 10`
- `Fail (0)` if `G3 < 10`

Dataset file:

- `archive/student-mat.csv`

## Visual Output

The script generates and saves these visualizations inside the `graphs/` folder:

- `g3_distribution.png`
- `pass_fail_countplot.png`
- `correlation_heatmap.png`
- `studytime_vs_result.png`
- `failures_vs_result.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `decision_tree.png`

## Preview

| Feature Importance | Decision Tree |
|---|---|
| ![Feature Importance](graphs/feature_importance.png) | ![Decision Tree](graphs/decision_tree.png) |

| Grade Distribution | Confusion Matrix |
|---|---|
| ![Grade Distribution](graphs/g3_distribution.png) | ![Confusion Matrix](graphs/confusion_matrix.png) |

## Project Structure

```text
.
|-- README.md
|-- Project_Report.md
|-- academic_report.md
|-- student_performance_analysis.py
|-- archive/
|   |-- student-mat.csv
|   |-- student-por.csv
|   `-- student.txt
`-- graphs/
    |-- confusion_matrix.png
    |-- correlation_heatmap.png
    |-- decision_tree.png
    |-- feature_importance.png
    |-- failures_vs_result.png
    |-- g3_distribution.png
    |-- pass_fail_countplot.png
    `-- studytime_vs_result.png
```

## How To Run

Run the script from the project folder:

```bash
python -u "student_performance_analysis.py"
```

If you are outside the project folder, use the full file path.

## Key Learning Outcomes

- Applied preprocessing and label encoding on mixed-type educational data
- Built and evaluated a classification model using scikit-learn
- Interpreted model behavior through feature importance and tree visualization
- Generated academic-report-friendly visuals from Python

## Repository Description

Data mining project using the UCI Student Performance Dataset to predict student pass/fail outcomes with a Decision Tree classifier, EDA, visualizations, and an academic report.

## Author

Mayank Pandey
