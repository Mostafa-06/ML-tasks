# Diabetes Prediction

Binary classification project that predicts whether a patient is **diabetic** from basic clinical measurements: age, gender, blood sugar level, creatinine, BMI, urea, cholesterol, and HbA1c. The notebook covers data cleaning, exploratory analysis, preprocessing, a comparison of 15 classifiers, 5-fold cross-validation, feature importance, and a small Gradio demo app.

> **Disclaimer:** this is an educational project. It is not a medical tool and must not be used for diagnosis.

## Dataset

- **Size:** 10,000 rows × 10 columns (9 after dropping `Patient_ID`), reduced to **9,932 rows** after dropping rows with missing values
- **Target:** `Class` (Diabetic / not diabetic)
- **Features:**

| Column | Type |
|---|---|
| `Age`, `BMI`, `Creatinine`, `Urea`, `Cholesterol` | numeric |
| `Blood_Sugar_Level`, `HbA1c` | numeric |
| `Gender` | categorical |

- **Data quality:** 297 missing values in total (no duplicates), and no outliers by the IQR rule

> The CSV is not included in this repo. Place it next to the notebook as `diabetes_patients_10000_full.csv`.

## Workflow

1. **Cleaning**: drop `Patient_ID` and rows with missing values
2. **EDA**: count plots, box plots, histograms, correlation heatmap, and IQR outlier check
3. **Train/test split**: 80/20, `random_state=42` (7,945 train / 1,987 test)
4. **Preprocessing**: label encoding for `Gender` and the target, standard scaling for numeric features
5. **Modeling**: 13 individual models, 1 soft-voting classifier, and 1 stacking classifier
6. **Evaluation**: confusion matrix, classification report, ROC curve, and precision-recall curve for every model
7. **Cross-validation**: 5-fold CV for LightGBM, CatBoost, XGBoost, and Random Forest
8. **Feature importance**: from a Random Forest
9. **Demo**: a Gradio interface for single-patient predictions

## Models Compared

| Family | Models |
|---|---|
| Linear | Logistic Regression |
| Distance-based | KNN |
| Kernel | SVC |
| Tree-based | Decision Tree |
| Probabilistic | Gaussian Naive Bayes |
| Bagging | Random Forest, Extra Trees |
| Boosting | AdaBoost, Gradient Boosting, Hist Gradient Boosting, XGBoost, CatBoost, LightGBM |
| Ensembles | Soft Voting (LR + RF + NB + XGB), Stacking (LR + RF + NB + LightGBM → XGBoost) |

## Results

Test-set performance for the diabetic class:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Tree and boosting models** (Decision Tree, Random Forest, CatBoost, XGBoost, LightGBM, Gradient Boosting, Hist GB, AdaBoost) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Voting Classifier | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Stacking Classifier | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Naive Bayes | 0.992 | 1.000 | 0.891 | 0.942 | 0.999 |
| Extra Trees | 0.987 | 0.984 | 0.837 | 0.904 | 0.999 |
| SVC | 0.973 | 0.830 | 0.796 | 0.812 | 0.994 |
| Logistic Regression | 0.965 | 0.781 | 0.728 | 0.754 | 0.989 |
| KNN | 0.956 | 0.750 | 0.612 | 0.674 | 0.957 |

5-fold cross-validation gives an accuracy and F1 of 1.00 for LightGBM, CatBoost, XGBoost, and Random Forest.

**Feature importance (Random Forest)**

| Feature | Importance |
|---|---|
| HbA1c | 0.485 |
| Blood_Sugar_Level | 0.440 |
| All other features | ≈ 0.015 each or less |

**Takeaways**

- Many models reach a perfect score, and cross-validation agrees. Together with the feature importances, this shows the label is almost fully determined by **HbA1c and blood sugar level**, which together account for about 92% of the importance.
- This is typical of a clean or synthetic dataset, where the class is defined by clinical thresholds. Perfect scores here should not be expected on real patient data.
- Models that cannot capture simple threshold rules easily (Logistic Regression, KNN, SVC) score noticeably lower.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Gradio · Matplotlib · Seaborn

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost gradio tqdm jupyter

# 3. Add the dataset as "diabetes_patients_10000_full.csv" in the project folder
# 4. Update the path in the first data-loading cell, then run:
jupyter notebook diabetes_patients.ipynb
```

The notebook was written in Google Colab. If you run it there, upload the CSV to `/content/`.

## Project Structure

```
├── diabetes_patients.ipynb                 # Full analysis, modeling, and demo
├── diabetes_patients_10000_full.csv        # Dataset (add separately)
└── README.md
```

## Possible Improvements

- Test on a real, noisier dataset (for example the Pima Indians Diabetes dataset)
- Check whether the perfect scores come from how the labels were generated
- Hyperparameter tuning and more cross-validation on all models
- SHAP analysis for individual predictions
- Deploy the Gradio demo (for example on Hugging Face Spaces)

## Author

**Mostafa**, Computer Science student (Scientific Computing)
