# Used Car Price Prediction

Regression project that predicts the **price of a used car** from its brand, age, mileage, engine size, fuel type, transmission, and number of previous owners. The notebook covers data cleaning, exploratory analysis, outlier removal, preprocessing, and a comparison of regression models, including voting and stacking ensembles.

## Dataset

- **Size:** 1,000 rows × 8 columns, reduced to **874 rows** after dropping rows with missing values (about 836 after removing price outliers)
- **Target:** `Price`
- **Features:**

| Column | Type | Notes |
|---|---|---|
| `Brand` | categorical | Kia, Mercedes, Nissan, BMW, ... |
| `Year` | numeric | 2010 – 2023 |
| `Mileage` | numeric | 5,207 – 199,812 |
| `Engine_Size` | numeric | 1.0 – 4.0 L |
| `Fuel_Type` | categorical | Petrol, Diesel, Electric, ... |
| `Transmission` | categorical | Manual / Automatic |
| `Previous_Owners` | numeric | Stored as text, converted to numeric |

> The CSV is not included in this repo. Place it next to the notebook as `cars_data.csv`.

## Workflow

1. **Cleaning**: drop rows with missing values (no duplicates found)
2. **EDA**: box plots and histograms for numeric columns, a correlation heatmap, and a Mileage vs Price scatter plot
3. **Outlier removal**: 38 `Price` outliers removed with the IQR method (1.5 × IQR)
4. **Preprocessing**:
   - Label encoding for `Brand`, `Transmission`, and `Fuel_Type`
   - `Previous_Owners` converted to numeric and filled with the training median
   - Standard scaling (fit on train only)
5. **Train/test split**: 90/10, `random_state=42`
6. **Modeling**: Linear baseline, 7 more regressors, a voting regressor, and a stacking regressor
7. **Evaluation**: R², MAE, and RMSE on train and test sets, plus an actual-vs-predicted plot

## Models Compared

| Family | Models |
|---|---|
| Linear | Linear Regression, Ridge, Lasso |
| Bagging | Random Forest |
| Boosting | Gradient Boosting, XGBoost, LightGBM, CatBoost |
| Ensembles | Voting (Linear + RF + XGB + CatBoost), Stacking |

## Results

Test-set performance, ranked by R²:

| Model | R² Test | MAE Test | RMSE Test | R² Train |
|---|---|---|---|---|
| **CatBoost** | 0.905 | 870 | 1,292 | 0.994 |
| LightGBM | 0.870 | 1,030 | 1,513 | 0.979 |
| XGBoost | 0.859 | 1,101 | 1,572 | 1.000 |
| Gradient Boosting | 0.858 | 1,122 | 1,581 | 0.936 |
| Voting Regressor | 0.837 | 1,153 | 1,692 | 0.950 |
| Random Forest | 0.742 | 1,488 | 2,131 | 0.968 |
| Linear / Ridge / Lasso | 0.525 | 2,228 | 2,887 | 0.485 |
| Stacking Regressor | -0.664 | 4,429 | 5,406 | -0.515 |

**Takeaways**

- CatBoost is the best model, with R² ≈ 0.91 and an average error of about 870 on the test set.
- Boosting models clearly beat the linear models, which suggests non-linear relationships between the features and the price.
- Most tree models have a large train/test gap (XGBoost: R² of 1.00 vs 0.86), which indicates overfitting on this small dataset.
- The stacking regressor performs worse than predicting the mean, which points to a configuration issue that should be investigated.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost tqdm jupyter

# 3. Add the dataset as "cars_data.csv" in the project folder
# 4. Update the path in the first data-loading cell, then run:
jupyter notebook cars_data.ipynb
```

The notebook was written in Google Colab. If you run it there, upload the CSV to `/content/`.

## Project Structure

```
├── cars_data.ipynb   # Full analysis and modeling
├── cars_data.csv     # Dataset (add separately)
└── README.md
```

## Possible Improvements

- Impute missing values instead of dropping about 13% of the rows
- Cross-validation, since the test set is only about 85 cars
- Hyperparameter tuning (GridSearchCV / Optuna) to reduce overfitting
- Fix the stacking regressor configuration
- One-hot encoding for `Brand` and `Fuel_Type`
- Feature importance and SHAP analysis

## Author

**Mostafa**, Computer Science student (Scientific Computing)
