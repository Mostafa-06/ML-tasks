# Student Performance Prediction

Regression project that predicts a student's **Performance Index** (10 – 100) from study habits and past results: hours studied, previous scores, extracurricular activities, sleep hours, and sample question papers practiced. The notebook covers data cleaning, exploratory analysis, preprocessing, and a comparison of 19 regression models, including voting and stacking ensembles.

## Dataset

- **Source:** [Student Performance (Multiple Linear Regression)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression) (Kaggle)
- **Size:** 10,000 rows × 6 columns, reduced to **9,868 rows** after removing 132 duplicates
- **Target:** `Performance Index` (mean ≈ 55, range 10 – 100)
- **Features:**

| Column | Type | Notes |
|---|---|---|
| `Hours Studied` | numeric | 1 – 9 |
| `Previous Scores` | numeric | 40 – 99 |
| `Extracurricular Activities` | categorical (Yes / No) | About 16% missing |
| `Sleep Hours` | numeric | 4 – 9 |
| `Sample Question Papers Practiced` | numeric | 0 – 9, about 29% missing |

> The CSV is not included in this repo. Place it next to the notebook as `Student_Performance.csv`.

## Workflow

1. **Cleaning**: remove duplicate rows
2. **EDA**: distribution of the target, Previous Scores vs Performance Index scatter plot, box plots, and a correlation heatmap
3. **Train/test split**: 80/20, `random_state=42` (7,894 train / 1,974 test)
4. **Preprocessing** (fit on train only):
   - Mean imputation for `Sample Question Papers Practiced`
   - Most-frequent imputation for `Extracurricular Activities`
   - Label encoding for `Extracurricular Activities`
   - Standard scaling for the numeric features
5. **Modeling**: Linear Regression baseline, 15 individual models, 3 voting regressors, and 1 stacking regressor
6. **Evaluation**: MAE, MSE, RMSE, and R² on train and test sets, plus actual-vs-predicted plots

## Models Compared

| Family | Models |
|---|---|
| Linear | Linear Regression, Lasso, Ridge |
| Distance-based | KNN |
| Kernel | SVR (linear kernel) |
| Tree-based | Decision Tree |
| Bagging | Random Forest, Extra Trees, Bagging |
| Boosting | AdaBoost, Gradient Boosting, Hist Gradient Boosting, XGBoost, CatBoost, LightGBM |
| Ensembles | Voting (LR + RF + DT), Weighted Voting, Stacking (RF + GB + CatBoost + LightGBM → XGBoost) |

## Results

Test-set performance, ranked by R²:

| Model | R² Test | RMSE Test | MAE Test | R² Train |
|---|---|---|---|---|
| **SVR (linear)** | 0.9886 | 2.075 | 1.652 | 0.9883 |
| Linear Regression | 0.9886 | 2.076 | 1.653 | 0.9883 |
| Ridge | 0.9886 | 2.076 | 1.653 | 0.9883 |
| CatBoost | 0.9878 | 2.152 | 1.721 | 0.9884 |
| LightGBM | 0.9877 | 2.156 | 1.726 | 0.9898 |
| Hist Gradient Boosting | 0.9877 | 2.156 | 1.726 | 0.9898 |
| Gradient Boosting | 0.9876 | 2.162 | 1.724 | 0.9884 |
| XGBoost | 0.9874 | 2.185 | 1.745 | 0.9907 |
| Bagging | 0.9861 | 2.292 | 1.830 | 0.9925 |
| Random Forest | 0.9861 | 2.294 | 1.832 | 0.9925 |
| Decision Tree | 0.9822 | 2.596 | 2.064 | 0.9912 |
| Extra Trees | 0.9812 | 2.664 | 2.135 | 0.9991 |
| Lasso | 0.9804 | 2.723 | 2.178 | 0.9802 |
| KNN | 0.9718 | 3.264 | 2.614 | 0.9812 |
| AdaBoost | 0.9624 | 3.773 | 2.995 | 0.9628 |

Ensemble R² on the test set: Stacking 0.9865, Voting 0.9850, Weighted Voting 0.9850.

**Takeaways**

- Simple **linear models are the best** (R² ≈ 0.989, average error of about 1.65 points). The relationship between the features and the performance index is almost linear, so complex models add no value.
- Tree-based models overfit slightly (Extra Trees: R² of 0.999 on train vs 0.981 on test) and score a little lower than the linear ones.
- The voting and stacking ensembles did not beat the linear baseline.
- Because the data looks almost linear, the model gives a useful reading of which habits matter most (for example, through the linear coefficients).

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost tqdm jupyter

# 3. Add the dataset as "Student_Performance.csv" in the project folder
# 4. Update the path in the first data-loading cell, then run:
jupyter notebook Student_Performance.ipynb
```

The notebook was written in Google Colab. If you run it there, upload the CSV to `/content/`.

## Project Structure

```
├── Student_Performance.ipynb   # Full analysis and modeling
├── Student_Performance.csv     # Dataset (add separately)
└── README.md
```

## Possible Improvements

- Inspect the linear model coefficients to rank the most important habits
- Cross-validation and a residual analysis
- Hyperparameter tuning for the tree-based models
- A small web app that predicts a score from study habits

## Author

**Mostafa**, Computer Science student (Scientific Computing)
