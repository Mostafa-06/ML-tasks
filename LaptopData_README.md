# Laptop Price Prediction

Regression project that predicts the **price of a laptop** from its specs: brand, type, screen size and resolution, CPU, RAM, storage, GPU, operating system, and weight. The notebook covers data cleaning, feature encoding, extra visualizations, and a comparison of 19 regression models, including voting and stacking ensembles.

## Dataset

- **Size:** 1,303 raw rows × 11 columns, reduced to **1,244 rows** after removing 58 duplicates and rows with missing values
- **Target:** `Price` (mean ≈ 59,956, range ≈ 9,271 – 324,955, right-skewed)
- **Features:**

| Column | Type | Handling |
|---|---|---|
| `Company`, `TypeName`, `OpSys` | categorical | One-hot encoding |
| `ScreenResolution`, `Cpu`, `Memory`, `Gpu` | high-cardinality text | Binary encoding |
| `Ram`, `Weight` | text with units ("8GB", "1.37kg") | Units stripped and converted to numeric |
| `Inches` | text | Converted to numeric, median imputation |

> The CSV is not included in this repo. Place it next to the notebook as `laptopData.csv`.

## Workflow

1. **Cleaning**: drop the index column and duplicates, drop rows with missing values
2. **Parsing**: extract numeric values from `Ram` and `Weight`, convert `Inches`, and impute missing values with the median
3. **Encoding**:
   - One-hot encoding for `Company`, `TypeName`, and `OpSys`
   - Binary encoding (`category_encoders`) for `ScreenResolution`, `Cpu`, `Memory`, and `Gpu`
4. **Train/test split**: 80/20, `random_state=100` (995 train / 249 test, 63 features)
5. **Scaling**: standard scaling
6. **EDA visualizations**: price distribution, laptops per brand, price by type, price by RAM, average price by OS, and price vs weight
7. **Modeling**: Ridge baseline, 15 individual models, 3 voting regressors, and 1 stacking regressor
8. **Evaluation**: MAE, MSE, RMSE, and R² on train and test sets

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

Test-set performance, ranked by R²:

| Model | R² Test | RMSE Test | MAE Test | R² Train |
|---|---|---|---|---|
| **LightGBM** | 0.788 | 16,802 | 10,635 | 0.936 |
| Hist Gradient Boosting | 0.783 | 16,969 | 10,932 | 0.940 |
| CatBoost | 0.771 | 17,430 | 11,246 | 0.900 |
| XGBoost | 0.768 | 17,553 | 10,734 | 0.973 |
| Bagging | 0.764 | 17,716 | 10,815 | 0.946 |
| Random Forest | 0.761 | 17,834 | 10,788 | 0.948 |
| Gradient Boosting | 0.751 | 18,177 | 11,936 | 0.884 |
| Extra Trees | 0.747 | 18,332 | 11,720 | 0.999 |
| KNN | 0.739 | 18,610 | 11,869 | 0.841 |
| Ridge / Lasso / Linear | 0.703 – 0.705 | ≈ 19,800 | ≈ 14,200 | 0.719 |
| Decision Tree | 0.611 | 22,736 | 13,286 | 0.965 |

Ensemble R² on the test set: Stacking 0.755, Voting 0.750, Weighted Voting 0.739.

**Takeaways**

- Gradient boosting models lead with R² ≈ 0.77 – 0.79, and the best average error is about 10,600.
- The linear models are a reasonable baseline (R² ≈ 0.70), so a lot of the price is explained by simple spec effects such as RAM and brand.
- The tree models show clear overfitting (Extra Trees: R² of 1.00 on train vs 0.75 on test), which is expected on only about 1,000 training rows.
- The voting and stacking ensembles did not beat the best single boosting model.

## Tech Stack

Python · pandas · NumPy · scikit-learn · category_encoders · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn category_encoders xgboost lightgbm catboost tqdm jupyter

# 3. Add the dataset as "laptopData.csv" in the project folder
# 4. Update the path in the first data-loading cell, then run:
jupyter notebook Laptop_csv.ipynb
```

## Project Structure

```
├── Laptop_csv.ipynb   # Full analysis and modeling
├── laptopData.csv     # Dataset (add separately)
└── README.md
```

## Possible Improvements

- Log-transform the skewed `Price` target
- Extract structured features from the text columns (CPU brand and speed, storage type and size, touchscreen, IPS, resolution in pixels)
- Hyperparameter tuning (GridSearchCV / Optuna) and cross-validation
- Feature importance and SHAP analysis
- A working demo app (for example with Gradio) that takes laptop specs and returns the predicted price

## Author

**Mostafa**, Computer Science student (Scientific Computing)
