# Ames Housing Price Prediction

Regression project that predicts the **sale price of houses** in Ames, Iowa from 80 property features. The notebook covers data exploration, outlier analysis, preprocessing, and a comparison of 19 regression models, including voting and stacking ensembles.

## Dataset

- **Source:** [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Kaggle / De Cock, 2011)
- **Size:** 1,460 rows × 81 columns
- **Target:** `SalePrice`
- **Features:** a mix of numeric and categorical attributes describing lot size, build quality, year built, basement, garage, neighborhood, and more
- **Missing values:** 7,829 in total, with no duplicate rows

> The CSV is not included in this repo. Download it and place it next to the notebook as `Ames Housing Dataset.csv`.

## Workflow

1. **EDA**: count plots for every categorical column, box plots and histograms (with KDE) for every numeric column
2. **Outlier detection**: IQR method (1.5 × IQR) per numeric column
3. **Missing values**: mode for categorical columns, mean for numeric columns
4. **Train/test split**: 80/20, `random_state=42` (1,168 train / 292 test)
5. **Preprocessing**:
   - Label encoding for categorical features
   - Standard scaling for numeric features (fit on train only)
6. **Modeling**: 15 individual models, 3 voting regressors, and 1 stacking regressor
7. **Evaluation**: MAE, MSE, RMSE, and R² on both train and test sets, plus actual-vs-predicted plots
8. **Dimensionality reduction**: PCA (3 components) was explored

## Models Compared

| Family | Models |
|---|---|
| Linear | Linear Regression, Lasso, Ridge |
| Distance-based | KNN |
| Kernel | SVR |
| Tree-based | Decision Tree |
| Bagging | Random Forest, Extra Trees, Bagging |
| Boosting | AdaBoost, Gradient Boosting, Hist Gradient Boosting, XGBoost, CatBoost, LightGBM |
| Ensembles | Voting (LR + RF + DT), Weighted Voting, Stacking (RF + GB + CatBoost + LightGBM → XGBoost) |

## Results

Top models on the held-out test set, ranked by R²:

| Model | R² Test | RMSE Test | MAE Test | R² Train |
|---|---|---|---|---|
| **XGBoost** | 0.911 | 26,065 | 16,435 | 0.996 |
| CatBoost | 0.902 | 27,428 | 16,975 | 0.959 |
| Gradient Boosting | 0.901 | 27,553 | 17,015 | 0.969 |
| Hist Gradient Boosting | 0.893 | 28,634 | 16,549 | 0.980 |
| Bagging | 0.890 | 29,083 | 18,044 | 0.973 |
| Extra Trees | 0.885 | 29,648 | 17,823 | 1.000 |
| LightGBM | 0.885 | 29,660 | 16,965 | 0.980 |
| Random Forest | 0.883 | 29,963 | 18,066 | 0.975 |
| Stacking | 0.879 | 29,660 | 16,965 | 0.981 |
| Linear Regression | 0.821 | 37,030 | 20,634 | 0.875 |
| Decision Tree | 0.784 | 40,701 | 27,153 | 0.988 |

**Takeaways**

- Gradient boosting methods lead, with XGBoost the best overall (R² ≈ 0.91, average error ≈ $16K).
- Tree-based models show a clear train/test gap (Extra Trees: R² of 1.00 on train vs 0.885 on test), which indicates overfitting.
- The linear models (Linear, Ridge, Lasso) are a solid baseline at R² ≈ 0.82 but cannot match the tree ensembles.
- The voting and stacking ensembles did not beat the best single boosting model here.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost tqdm jupyter

# 3. Add the dataset as "Ames Housing Dataset.csv" in the project folder
# 4. Update the path in the first data-loading cell, then run:
jupyter notebook Ames_Housing.ipynb
```

The notebook was written in Google Colab. If you run it there, upload the CSV to `/content/`.

## Project Structure

```
├── Ames_Housing.ipynb         # Full analysis and modeling
├── Ames Housing Dataset.csv   # Dataset (download separately)
└── README.md
```

## Possible Improvements

- Hyperparameter tuning (GridSearchCV / Optuna) for the top boosting models
- K-fold cross-validation instead of a single split
- Log-transform the skewed `SalePrice` target
- Feature engineering (e.g. total square footage, house age)
- One-hot or ordinal encoding instead of label encoding
- Feature importance and SHAP analysis

## Author

**Mostafa**, Computer Science student (Scientific Computing)
