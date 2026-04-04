# Student Performance Prediction Using Decision Tree

This is my Data Mining and Data Warehousing project based on the UCI Student Performance dataset.

In this project, I used the `student-mat.csv` dataset and built a Decision Tree model to predict whether a student will pass or fail based on different factors like study time, past failures, family background, absences, and previous grades.

## What this project does

- loads the student dataset
- checks and preprocesses the data
- encodes categorical columns
- creates a pass/fail target using `G3`
- trains a Decision Tree classifier
- evaluates the model using accuracy, confusion matrix, and classification report
- saves graphs in the `graphs/` folder

## Main result

- Accuracy: `86.55%`
- Dataset size: `395` records
- Features used for training: `32`
- Model used: Decision Tree Classifier
- Target:
  `Pass = G3 >= 10`
  `Fail = G3 < 10`

## Files in this repo

- `student_performance_analysis.py` - main Python file
- `archive/student-mat.csv` - dataset used in the project
- `graphs/` - all generated graphs
- `requirements.txt` - libraries needed to run the project
- `Student_Performance_Project_Report_dm.docx` - final report

## Graphs generated

- `g3_distribution.png`
- `pass_fail_countplot.png`
- `correlation_heatmap.png`
- `studytime_vs_result.png`
- `failures_vs_result.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `decision_tree.png`

## How to run

Install the required libraries:

```bash
python3 -m pip install -r requirements.txt
```

Then run the project:

```bash
python3 student_performance_analysis.py
```

## Tools used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Note

This project is mainly for academic learning and to understand how data mining can be used for student performance prediction.
