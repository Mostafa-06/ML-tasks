# Bank Marketing: Term Deposit Prediction

Binary classification project that predicts whether a bank client will **subscribe to a term deposit** (`deposit` = yes / no) based on client demographics, financial status, and details of the marketing campaign. The notebook covers data cleaning, exploratory analysis, preprocessing, and a comparison of 15 classifiers, including voting and stacking ensembles.

## Dataset

- **Size:** 8,912 raw rows × 17 columns, reduced to **8,719 rows** after dropping rows with missing values
- **Target:** `deposit` (yes / no), fairly balanced classes (about 46% yes in the training set)
- **Features:**

| Group | Columns |
|---|---|
| Client | `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan` |
| Last contact | `contact`, `day`, `month`, `duration` |
| Campaign history | `campaign`, `pdays`, `previous`, `poutcome` (dropped) |

> The CSV is not included in this repo. Place it next to the notebook as `bank.csv`.

## Workflow

1. **Cleaning**: drop rows with missing values, reset the index, and drop the `poutcome` column
2. **EDA**: count plots for categorical columns, box plots, histograms (with KDE), and a correlation heatmap
3. **Outlier detection**: IQR method (1.5 × IQR) per numeric column
4. **Train/test split**: 90/10, `random_state=42` (7,847 train / 872 test)
5. **Preprocessing**:
   - Label encoding for categorical features and the target
   - Standard scaling for all features
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

Test-set performance, ranked by F1 score for the `yes` class:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **CatBoost** | 0.880 | 0.839 | 0.912 | 0.874 | 0.937 |
| LightGBM | 0.878 | 0.836 | 0.915 | 0.874 | 0.935 |
| Hist Gradient Boosting | 0.873 | 0.831 | 0.907 | 0.867 | 0.933 |
| XGBoost | 0.869 | 0.828 | 0.902 | 0.864 | 0.929 |
| Random Forest | 0.860 | 0.814 | 0.900 | 0.855 | 0.931 |
| Stacking Classifier | 0.860 | 0.814 | 0.900 | 0.855 | 0.921 |
| Voting Classifier | 0.860 | 0.825 | 0.882 | 0.853 | 0.918 |
| Extra Trees | 0.859 | 0.834 | 0.865 | 0.849 | 0.919 |
| Gradient Boosting | 0.852 | 0.807 | 0.890 | 0.847 | 0.923 |
| SVC | 0.811 | 0.768 | 0.843 | 0.803 | 0.898 |
| AdaBoost | 0.811 | 0.782 | 0.815 | 0.798 | 0.890 |
| Logistic Regression | 0.802 | 0.787 | 0.777 | 0.782 | 0.867 |
| Decision Tree | 0.786 | 0.766 | 0.767 | 0.767 | 0.784 |
| Naive Bayes | 0.749 | 0.683 | 0.845 | 0.755 | 0.819 |
| KNN | 0.779 | 0.773 | 0.733 | 0.752 | 0.855 |

**Takeaways**

- Gradient boosting models (CatBoost, LightGBM, HistGB) lead with about 88% accuracy and an ROC-AUC of about 0.94.
- Random Forest, Extra Trees, and Decision Tree reach 100% train accuracy, which indicates overfitting.
- The ensembles (voting and stacking) did not beat the best single boosting models.
- Logistic Regression is a reasonable baseline at about 80% accuracy.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost tqdm jupyter

# 3. Add the dataset as "bank.csv" in the project folder
# 4. Update the path in the first data-loading cell, then run:
jupyter notebook bank.ipynb
```

The notebook was written in Google Colab. If you run it there, upload the CSV to `/content/`.

## Project Structure

```
├── bank.ipynb     # Full analysis and modeling
├── bank.csv       # Dataset (add separately)
└── README.md
```

## Possible Improvements

- Hyperparameter tuning (GridSearchCV / Optuna) and cross-validation
- Handle missing values by imputation instead of dropping rows
- Try removing `duration`: call length is only known after the call, so it can inflate results
- One-hot encoding for nominal features such as `job` and `month`
- Feature importance and SHAP analysis

## Author

**Mostafa**, Computer Science student (Scientific Computing)
