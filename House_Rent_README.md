# House Rent Prediction (India)

Regression project that predicts the **monthly rent of houses and apartments** across six major Indian cities from listing features such as size, BHK, floor, city, furnishing status, and locality. The notebook covers data cleaning, exploratory analysis, feature engineering, preprocessing, and a comparison of 19 regression models, including voting and stacking ensembles.

## Dataset

- **Source:** [House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset) (Kaggle)
- **Size:** 4,746 rows × 12 columns (4,741 rows after dropping missing `Rent`)
- **Target:** `Rent` (INR per month)
- **Features:**

| Column | Type | Notes |
|---|---|---|
| `BHK`, `Bathroom` | numeric | Bedrooms and bathrooms |
| `Size` | numeric | Area in sq. ft. |
| `Floor` | text | e.g. "Ground out of 2", split into `Floor_Num` and `Total_Floors` |
| `Area Type` | categorical | Super Area, Carpet Area, Built Area |
| `Area Locality` | categorical | About 2,200 unique localities |
| `City` | categorical | 6 cities |
| `Furnishing Status` | categorical | Furnished, Semi-Furnished, Unfurnished |
| `Tenant Preferred` | categorical | Bachelors, Family, or both |
| `Point of Contact` | categorical | Owner, Agent, Builder |
| `Posted On` | date | Dropped |

> The CSV is not included in this repo. Place it next to the notebook as `House_Rent_Dataset.csv`.

## Workflow

1. **Cleaning**: drop `Posted On`, drop rows with missing `Rent`
2. **EDA**: box plots, rent distribution, Size vs Rent scatter plot, and a correlation heatmap
3. **Outlier handling**: IQR clipping (winsorizing) on all numeric columns
4. **Feature engineering**: parse `Floor` into `Floor_Num` (Ground = 0, basements negative) and `Total_Floors`
5. **Encoding**:
   - One-hot encoding for `Area Type`, `City`, `Furnishing Status`, `Tenant Preferred`, and `Point of Contact`
   - Ordinal encoding for `Area Locality` (unseen localities mapped to `-1`)
6. **Train/test split**: 80/20, `random_state=42`, then drop remaining NaN and duplicate rows (3,768 train / 947 test)
7. **Scaling**: standard scaling for `Size`
8. **Modeling**: Linear Regression baseline, 15 individual models, 3 voting regressors, and 1 stacking regressor
9. **Evaluation**: MAE, MSE, RMSE, and R² on train and test sets, plus actual-vs-predicted plots

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

Test-set performance, ranked by R² (errors in INR):

| Model | R² Test | RMSE Test | MAE Test | R² Train |
|---|---|---|---|---|
| **LightGBM** | 0.826 | 8,344 | 5,187 | 0.925 |
| CatBoost | 0.825 | 8,375 | 5,201 | 0.881 |
| Hist Gradient Boosting | 0.823 | 8,420 | 5,253 | 0.925 |
| XGBoost | 0.821 | 8,470 | 5,249 | 0.947 |
| Extra Trees | 0.816 | 8,591 | 5,356 | 1.000 |
| Bagging | 0.815 | 8,613 | 5,585 | 0.942 |
| Random Forest | 0.809 | 8,752 | 5,567 | 0.942 |
| Linear Regression | 0.767 | 9,664 | 6,899 | 0.787 |
| Decision Tree | 0.731 | 10,383 | 6,124 | 0.941 |
| KNN | 0.552 | 13,396 | 9,096 | 0.760 |

Ensemble R² on the test set: Voting 0.810, Weighted Voting 0.805, Stacking 0.796.

**Takeaways**

- Gradient boosting models (LightGBM, CatBoost, HistGB, XGBoost) perform best, with R² ≈ 0.82 and an average error of about ₹5,200.
- Linear models are a decent baseline (R² ≈ 0.77) but cannot capture the non-linear patterns.
- Extra Trees reaches 100% train R², and most tree models have a clear train/test gap, which indicates overfitting.
- The voting and stacking ensembles did not beat the best single boosting model.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost tqdm jupyter

# 3. Add the dataset as "House_Rent_Dataset.csv" in the project folder
# 4. Update the path in the first data-loading cell, then run:
jupyter notebook House_Rent.ipynb
```

The notebook was written in Google Colab. If you run it there, upload the CSV to `/content/`.

## Project Structure

```
├── House_Rent.ipynb         # Full analysis and modeling
├── House_Rent_Dataset.csv   # Dataset (add separately)
└── README.md
```

## Possible Improvements

- Log-transform the skewed `Rent` target
- Target or frequency encoding for the high-cardinality `Area Locality`
- Hyperparameter tuning (GridSearchCV / Optuna) and cross-validation
- Feature importance and SHAP analysis

## Author

**Mostafa**, Computer Science student (Scientific Computing)
