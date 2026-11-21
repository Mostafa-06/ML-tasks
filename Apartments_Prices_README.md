# Apartment Prices Prediction (Egypt)

Regression project that predicts the **price of apartments for sale in Egypt (in EGP)** from listing features such as area, rooms, floor level, view, and finishing type. The notebook covers data cleaning, exploratory analysis, preprocessing, and a comparison of 19 regression models, including voting and stacking ensembles.

## Dataset

- **Size:** 5,578 raw listings × 11 columns, reduced to **4,089 rows** after dropping 1,489 duplicates
- **Target:** `price` (EGP)
- **Features:**

| Column | Type | Notes |
|---|---|---|
| `area_sqm` | numeric | Apartment area in m² |
| `rooms`, `bathrooms` | numeric | Parsed from text such as "2 rooms" |
| `floor_level` | numeric | "Ground" converted to 0 |
| `year_built` | numeric | Range 1915 – 2037 (includes under-construction projects) |
| `price_per_sqm` | numeric | Parsed from text such as "18,025 EGP/M²" |
| `seller_type` | categorical | Developer / Broker / ... |
| `view` | categorical | Main Street, Club, Lake, ... |
| `payment_method` | categorical | Cash, Installments, or both |
| `finishing_type` | categorical | Semi Finished, Super Lux, Extra Super Lux, ... |

> The CSV is not included in this repo. Place it next to the notebook as `Apartments Prices Dataset.csv`.

## Workflow

1. **Cleaning**: remove duplicates and convert text columns (`price`, `area_sqm`, `rooms`, `bathrooms`, `price_per_sqm`, `floor_level`) to numeric with a regex
2. **EDA**: count plots for categorical columns, box plots, histograms (with KDE), and a correlation heatmap
3. **Outlier detection**: IQR method (1.5 × IQR) per numeric column
4. **Train/test split**: 80/20, `random_state=42` (3,271 train / 818 test)
5. **Preprocessing**:
   - Label encoding for categorical features
   - Standard scaling
   - Median imputation for the few remaining missing values
6. **Modeling**: 15 individual models, 3 voting regressors, and 1 stacking regressor
7. **Evaluation**: MAE, MSE, RMSE, and R² on train and test sets, plus actual-vs-predicted plots
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

Test-set performance, ranked by R² (errors are in EGP):

| Model | R² Test | RMSE Test | MAE Test | R² Train |
|---|---|---|---|---|
| **Extra Trees** | 0.981 | 711,212 | 121,695 | 1.000 |
| Voting | 0.968 | 928,051 | 290,549 | 0.989 |
| Random Forest | 0.966 | 957,322 | 177,238 | 0.996 |
| Weighted Voting | 0.966 | 957,461 | 328,528 | 0.986 |
| Gradient Boosting | 0.963 | 993,424 | 248,208 | 0.994 |
| Bagging | 0.955 | 1,091,126 | 183,091 | 0.993 |
| Decision Tree | 0.945 | 1,206,457 | 322,892 | 0.998 |
| Ridge / Linear Regression | 0.931 | 1,353,952 | 695,869 | 0.919 |
| CatBoost | 0.883 | 1,764,450 | 344,983 | 0.985 |
| XGBoost | 0.868 | 1,878,801 | 230,773 | 0.998 |
| LightGBM | 0.708 | 2,792,323 | 350,201 | 0.910 |
| SVR (linear) | 0.032 | 5,079,500 | 2,750,146 | 0.039 |

**Takeaways**

- Tree ensembles (Extra Trees, Random Forest, Gradient Boosting) give the best accuracy.
- Linear models are a strong, stable baseline (R² ≈ 0.93) with almost no train/test gap.
- The big train/test gaps for XGBoost, CatBoost, and LightGBM point to overfitting with the default settings.
- The linear-kernel SVR performs poorly, likely because the target is very large and its range is wide.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost tqdm jupyter

# 3. Add the dataset as "Apartments Prices Dataset.csv" in the project folder
# 4. Update the path in the first data-loading cell, then run:
jupyter notebook Apartments_Prices.ipynb
```

The notebook was written in Google Colab. If you run it there, upload the CSV to `/content/`.

## Project Structure

```
├── Apartments_Prices.ipynb          # Full analysis and modeling
├── Apartments Prices Dataset.csv    # Dataset (add separately)
└── README.md
```

## Possible Improvements

- Handle outliers (e.g. extreme `rooms` and `price_per_sqm` values)
- Log-transform the skewed `price` target
- Hyperparameter tuning (GridSearchCV / Optuna) and cross-validation
- One-hot or ordinal encoding instead of label encoding
- Add location features (city or compound) if available
- Feature importance and SHAP analysis

## Author

**Mostafa**, Computer Science student (Scientific Computing)
