# Adult Census Income Prediction

Binary classification project that predicts whether a person earns **more than $50K per year** based on demographic and employment attributes from the UCI **Adult Census Income** dataset. The notebook covers the full workflow: data cleaning, exploratory analysis, preprocessing, and a comparison of 15 machine learning models.

## Dataset

- **Source:** [Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult) (UCI / Kaggle)
- **Size:** 32,561 rows × 15 columns (before removing duplicates)
- **Target:** `income` (`<=50K` or `>50K`)
- **Features:** age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours per week, native country, and more
- **Missing values:** marked as `?` in the raw file and loaded as `NaN`

> The CSV is not included in this repo. Download it and place it next to the notebook as `Adult Census Income.csv`.

## Workflow

1. **Data cleaning**: parse `?` as missing values, inspect nulls, and drop duplicate rows
2. **EDA**: income class distribution, correlation heatmap, and count plots for low-cardinality columns
3. **Train/test split**: 80/20, stratified on `income`, `random_state=42`
4. **Preprocessing** (fit on train only, to avoid data leakage):
   - Mode imputation for `workclass`, `occupation`, and `native.country`
   - Label encoding for categorical features (unseen test categories mapped to `-1`)
   - Standard scaling for numeric features
5. **Class imbalance**: SMOTE was explored on the training set
6. **Modeling**: baseline models, then a full comparison of 15 classifiers
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

Top models on the held-out test set, ranked by F1 score for the `>50K` class:

| Model | Accuracy | F1 | Balanced Acc. | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **LightGBM** | 0.867 | 0.699 | 0.790 | 0.920 | 0.818 |
| Hist Gradient Boosting | 0.866 | 0.696 | 0.787 | 0.919 | 0.816 |
| CatBoost | 0.866 | 0.695 | 0.786 | 0.920 | 0.817 |
| Stacking Classifier | 0.862 | 0.691 | 0.786 | 0.915 | 0.806 |
| XGBoost | 0.862 | 0.689 | 0.784 | 0.919 | 0.814 |
| Random Forest | 0.853 | 0.668 | 0.771 | 0.898 | 0.768 |
| Logistic Regression | 0.823 | 0.550 | 0.695 | 0.849 | 0.689 |

**Takeaways**

- Gradient boosting models (LightGBM, HistGB, CatBoost, XGBoost) consistently lead.
- Random Forest, Extra Trees, and Decision Tree reach 100% train accuracy, which points to overfitting.
- Simple models (Logistic Regression, Naive Bayes, SVC) have low recall on the `>50K` class because of the class imbalance.

## Tech Stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm catboost tqdm jupyter

# 3. Add the dataset as "Adult Census Income.csv" in the project folder
# 4. Update the path in the first data-loading cell, then run:
jupyter notebook Adult_Census.ipynb
```

The notebook was written in Google Colab. If you run it there, upload the CSV to `/content/`.

## Project Structure

```
├── Adult_Census.ipynb        # Full analysis and modeling
├── Adult Census Income.csv   # Dataset (download separately)
└── README.md
```

## Possible Improvements

- Hyperparameter tuning (GridSearchCV / Optuna) for the top boosting models
- Cross-validation instead of a single train/test split
- Decision-threshold tuning to improve recall on the `>50K` class
- One-hot or target encoding instead of label encoding for nominal features
- Feature importance and SHAP analysis

## Author

**Mostafa**, Computer Science student (Scientific Computing)
