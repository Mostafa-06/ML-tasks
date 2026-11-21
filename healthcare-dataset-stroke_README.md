# Stroke Prediction

Binary classification project that predicts whether a patient is likely to have a **stroke** based on demographic, lifestyle, and health attributes. The main challenge is **severe class imbalance**: only about 5% of patients in the data had a stroke. The notebook covers data cleaning, exploratory analysis, preprocessing, and a comparison of 15 classifiers.

> **Disclaimer:** this is an educational project. It is not a medical tool and must not be used for diagnosis.

## Dataset

- **Source:** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle)
- **Size:** 5,110 rows × 12 columns (11 after dropping `id`)
- **Target:** `stroke` (1 = stroke, 0 = no stroke), with **249 positive** vs 4,861 negative cases (about 4.9%)
- **Features:**

| Column | Type |
|---|---|
| `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status` | categorical |
| `age`, `avg_glucose_level`, `bmi` | numeric |
| `hypertension`, `heart_disease` | binary |

- **Missing values:** 201 missing `bmi` values

> The CSV is not included in this repo. Place it next to the notebook as `healthcare-dataset-stroke-data.csv`.

## Workflow

1. **Cleaning**: drop `id`, drop the single `Other` gender row
2. **EDA**: class distribution, count plots for low-cardinality columns, correlation heatmap, BMI distribution and skewness, box plots
3. **Train/test split**: 90/10, stratified on `stroke`, `random_state=42` (511 test rows, 25 of them positive)
4. **Preprocessing** (fit on train only):
   - Median imputation for `bmi`
   - Label encoding for categorical features
   - Standard scaling for `age`, `bmi`, and `avg_glucose_level`
5. **Baseline**: Logistic Regression with `class_weight='balanced'`
6. **Modeling**: 13 individual models, 1 soft-voting classifier, and 1 stacking classifier
7. **Evaluation**: confusion matrix, classification report, ROC curve, and precision-recall curve for every model

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

Because only 5% of cases are positive, **accuracy is misleading** (a model that always predicts "no stroke" scores about 95%). The table focuses on the stroke class, ranked by F1:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Naive Bayes | 0.873 | 0.155 | 0.360 | 0.217 | 0.765 | 0.165 |
| LightGBM | 0.947 | 0.375 | 0.120 | 0.182 | 0.801 | 0.224 |
| Voting Classifier | 0.945 | 0.333 | 0.120 | 0.176 | 0.777 | 0.193 |
| XGBoost | 0.937 | 0.231 | 0.120 | 0.158 | 0.773 | 0.154 |
| CatBoost | 0.947 | 0.333 | 0.080 | 0.129 | 0.797 | 0.211 |
| Logistic Regression | 0.951 | 0.000 | 0.000 | 0.000 | 0.800 | 0.251 |
| SVC / AdaBoost / KNN | ≈ 0.95 | 0.000 | 0.000 | 0.000 | 0.63 – 0.78 | 0.09 – 0.14 |

The balanced Logistic Regression baseline reaches a stroke **recall of 0.68** (17 of 25 cases caught), at the cost of lower precision (0.13) and accuracy (0.76).

**Takeaways**

- Most models reach about 95% accuracy only by predicting almost everyone as "no stroke", so their recall for strokes is very low.
- Class weighting helps recall a lot, which matters more than accuracy in a screening setting.
- Tree models reach 100% train scores, which indicates overfitting.
- With only 25 positive test cases, every metric is noisy, so differences between models should be read with care.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost tqdm jupyter

# 3. Add the dataset as "healthcare-dataset-stroke-data.csv" in the project folder
# 4. Update the path in the first data-loading cell, then run:
jupyter notebook healthcare-dataset-stroke.ipynb
```

The notebook was written in Google Colab. If you run it there, upload the CSV to `/content/`.

## Project Structure

```
├── healthcare-dataset-stroke.ipynb       # Full analysis and modeling
├── healthcare-dataset-stroke-data.csv    # Dataset (add separately)
└── README.md
```

## Possible Improvements

- Handle the imbalance for all models (class weights, SMOTE, or undersampling)
- Tune the decision threshold to trade precision for recall
- Use stratified k-fold cross-validation instead of a single small test set
- Optimize for recall, F1, or PR-AUC rather than accuracy
- Feature importance and SHAP analysis

## Author

**Mostafa**, Computer Science student (Scientific Computing)
